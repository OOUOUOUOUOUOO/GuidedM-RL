from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: float


class SACAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        device: str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        replay_size: int = 200000,
        batch_size: int = 256,
        start_steps: int = 1000,
        update_after: int = 1000,
        updates_per_step: int = 1,
    ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.updates_per_step = updates_per_step

        self.actor = Actor(obs_dim, action_dim, hidden_size).to(device)
        self.q1 = QNet(obs_dim, action_dim, hidden_size).to(device)
        self.q2 = QNet(obs_dim, action_dim, hidden_size).to(device)
        self.q1_target = QNet(obs_dim, action_dim, hidden_size).to(device)
        self.q2_target = QNet(obs_dim, action_dim, hidden_size).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic)

        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = float(target_entropy)

        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=device, requires_grad=True)
            self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=device)
            self.optimizer_alpha = None

        self.replay: Deque[Transition] = deque(maxlen=int(replay_size))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs, deterministic: bool = False):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            single = obs_t.ndim == 1
            if single:
                obs_t = obs_t.unsqueeze(0)

            action, logp = self._sample_action(obs_t, deterministic=deterministic)
            action_np = action.cpu().numpy()
            logp_np = logp.cpu().numpy()
            if single:
                return action_np[0], float(logp_np[0]), 0.0
            return action_np, logp_np, np.zeros((action_np.shape[0],), dtype=np.float32)

    def select_action(self, obs, deterministic: bool = False):
        return self.act(obs, deterministic=deterministic)

    def _sample_action(self, obs_t: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.actor(obs_t)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        z = mean if deterministic else dist.rsample()
        action = torch.tanh(z)
        eps = 1e-6
        logp = dist.log_prob(z).sum(dim=-1) - torch.log(1 - action.pow(2) + eps).sum(dim=-1)
        return action, logp

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=float(done),
            )
        )

    def can_update(self, global_step: int) -> bool:
        return global_step >= self.update_after and len(self.replay) >= self.batch_size

    def _sample_batch(self):
        batch = random.sample(self.replay, self.batch_size)
        obs = torch.as_tensor(np.array([b.obs for b in batch]), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.array([b.action for b in batch]), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(np.array([b.reward for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.as_tensor(np.array([b.next_obs for b in batch]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.array([b.done for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        return obs, action, reward, next_obs, done

    def update(self) -> Dict[str, float]:
        metrics = {"loss_q": 0.0, "loss_actor": 0.0, "alpha": float(self.alpha.detach().cpu().item())}

        for _ in range(self.updates_per_step):
            obs, action, reward, next_obs, done = self._sample_batch()

            with torch.no_grad():
                next_action, next_logp = self._sample_action(next_obs, deterministic=False)
                target_q1 = self.q1_target(next_obs, next_action)
                target_q2 = self.q2_target(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_logp.unsqueeze(1)
                backup = reward + self.gamma * (1.0 - done) * target_q

            q1 = self.q1(obs, action)
            q2 = self.q2(obs, action)
            loss_q = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

            self.optimizer_q.zero_grad()
            loss_q.backward()
            self.optimizer_q.step()

            new_action, logp = self._sample_action(obs, deterministic=False)
            q_pi = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
            loss_actor = (self.alpha.detach() * logp.unsqueeze(1) - q_pi).mean()

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            if self.auto_alpha and self.optimizer_alpha is not None:
                alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                self.optimizer_alpha.zero_grad()
                alpha_loss.backward()
                self.optimizer_alpha.step()

            with torch.no_grad():
                for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                    p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
                for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                    p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

            metrics["loss_q"] += float(loss_q.detach().cpu().item())
            metrics["loss_actor"] += float(loss_actor.detach().cpu().item())
            metrics["alpha"] = float(self.alpha.detach().cpu().item())

        n = float(max(self.updates_per_step, 1))
        metrics["loss_q"] /= n
        metrics["loss_actor"] /= n
        return metrics

    def save(self, path_prefix: str):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.q1.state_dict(), f"{path_prefix}_q1.pth")
        torch.save(self.q2.state_dict(), f"{path_prefix}_q2.pth")
        torch.save(self.q1_target.state_dict(), f"{path_prefix}_q1_target.pth")
        torch.save(self.q2_target.state_dict(), f"{path_prefix}_q2_target.pth")
        torch.save({"log_alpha": self.log_alpha.detach().cpu()}, f"{path_prefix}_alpha.pth")

    def load(self, path_prefix: str, map_location: Optional[str] = None):
        if map_location is None:
            map_location = self.device

        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth", map_location=map_location))
        self.q1.load_state_dict(torch.load(f"{path_prefix}_q1.pth", map_location=map_location))
        self.q2.load_state_dict(torch.load(f"{path_prefix}_q2.pth", map_location=map_location))

        q1_t_path = f"{path_prefix}_q1_target.pth"
        q2_t_path = f"{path_prefix}_q2_target.pth"
        try:
            self.q1_target.load_state_dict(torch.load(q1_t_path, map_location=map_location))
            self.q2_target.load_state_dict(torch.load(q2_t_path, map_location=map_location))
        except FileNotFoundError:
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())

        alpha_path = f"{path_prefix}_alpha.pth"
        try:
            alpha_state = torch.load(alpha_path, map_location=map_location)
            if isinstance(alpha_state, dict) and "log_alpha" in alpha_state:
                loaded = torch.as_tensor(alpha_state["log_alpha"], dtype=torch.float32, device=self.device)
                if self.auto_alpha:
                    self.log_alpha.data.copy_(loaded)
                else:
                    self.log_alpha = loaded
        except FileNotFoundError:
            pass

        self.eval()
        return self
