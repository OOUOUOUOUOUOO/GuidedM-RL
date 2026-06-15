# agents/ppo_agent.py
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_size=128,
        device="cpu",
        gamma=0.95,
        lam=0.97,
        clip_eps=0.2,
        # Value loss controls
        value_loss_coef=1.0,
        clip_vf_eps: Optional[float] = None,
        # Training stability
        max_grad_norm: Optional[float] = 0.5,
        target_kl: Optional[float] = None,
        entropy_coef=0.01,
        lr_actor=3e-4,
        lr_critic=3e-4,
        update_epochs=4,
        batch_size=64
    ):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_loss_coef = value_loss_coef
        self.clip_vf_eps = clip_vf_eps
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        ).to(device)
        self.log_std = nn.Parameter(torch.zeros(action_dim).to(device))

        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        ).to(device)

        self.optimizer_actor = torch.optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 经验池
        self.memory = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": []
        }

    def select_action(self, obs, deterministic: bool = False):
        """
        Unified action selection API.
        - deterministic=True: use mean action
        - deterministic=False: sample from Gaussian
        """
        return self.act(obs, deterministic=deterministic)

    def act(self, obs, deterministic: bool = False):
        # Inference-time only: avoid building autograd graphs.
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            mean = self.actor(obs_tensor)
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            z = mean if deterministic else dist.sample()  # pre-squash
            action = torch.tanh(z)

            # Squashed Gaussian log-prob via change of variables.
            # log p(a) = log p(z) - sum(log|da/dz|)
            # da/dz = 1 - tanh(z)^2 = 1 - a^2
            eps = 1e-6
            if action.ndim == 1:
                log_prob = dist.log_prob(z).sum() - torch.log(1 - action.pow(2) + eps).sum()
                value = self.critic(obs_tensor)
                return (
                    action.cpu().numpy(),
                    float(log_prob.cpu().item()),
                    float(value.cpu().item()),
                )

            log_prob = dist.log_prob(z).sum(dim=1) - torch.log(1 - action.pow(2) + eps).sum(dim=1)
            value = self.critic(obs_tensor).squeeze(1)

        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def store_transition(self, obs, action, reward, done, log_prob, value):
        self.memory["obs"].append(obs)
        self.memory["actions"].append(action)
        self.memory["log_probs"].append(log_prob)
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)
        self.memory["values"].append(value)

    def clear_memory(self):
        for k in self.memory.keys():
            self.memory[k] = []

    def evaluate(self, obs, action):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        # Clamp to avoid atanh singularities at exactly +/-1.
        eps = 1e-6
        action = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        z = self._atanh(action)

        log_prob_z = dist.log_prob(z).sum(dim=1, keepdim=True)
        log_det = torch.log(1 - action.pow(2) + eps).sum(dim=1, keepdim=True)
        log_probs = log_prob_z - log_det

        # True entropy of a squashed Gaussian is non-trivial; keep an approximation.
        entropy = dist.entropy().sum(dim=1, keepdim=True)
        value = self.critic(obs)
        return log_probs, entropy, value

    @staticmethod
    def _atanh(x, eps: float = 1e-6):
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def compute_gae(self, last_value: float = 0.0):
        rewards = self.memory["rewards"]
        values = self.memory["values"] + [float(last_value)]
        dones = self.memory["dones"]

        gae = 0
        advantages = []
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return advantages, returns

    def update(self, last_value: float = 0.0):
        if not self.memory["obs"]:
            return {}

        obs_batch = torch.tensor(self.memory["obs"], dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(self.memory["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(self.memory["log_probs"], dtype=torch.float32, device=self.device).unsqueeze(1)
        old_values = torch.tensor(self.memory["values"], dtype=torch.float32, device=self.device).unsqueeze(1)
        advantages, returns = self.compute_gae(last_value=last_value)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)

        dataset_size = len(obs_batch)
        # Basic metrics for training logs.
        loss_actor_sum = 0.0
        loss_critic_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        clip_frac_sum = 0.0
        num_minibatches = 0
        for _ in range(self.update_epochs):
            # Shuffle to reduce correlations between minibatches.
            perm = torch.randperm(dataset_size, device=self.device)
            early_stop = False

            for i in range(0, dataset_size, self.batch_size):
                idx = perm[i : i + self.batch_size]

                obs = obs_batch[idx]
                actions = action_batch[idx]
                old_logp = old_log_probs[idx]
                ret = returns[idx]
                adv = advantages[idx]
                old_val = old_values[idx]

                log_probs, entropy, values = self.evaluate(obs, actions)
                ratio = torch.exp(log_probs - old_logp)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                clip_frac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()

                loss_actor = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                # Value function loss (optionally clipped, like PPO2).
                if self.clip_vf_eps is not None and self.clip_vf_eps > 0:
                    values_clipped = old_val + torch.clamp(
                        values - old_val, -self.clip_vf_eps, self.clip_vf_eps
                    )
                    value_loss_unclipped = (values - ret).pow(2)
                    value_loss_clipped = (values_clipped - ret).pow(2)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = nn.MSELoss()(values, ret)

                loss_critic = self.value_loss_coef * value_loss

                # Early stopping by KL divergence (approx).
                if self.target_kl is not None and self.target_kl > 0:
                    approx_kl = (old_logp - log_probs).mean()
                    approx_kl_sum += float(approx_kl.detach().cpu().item())
                    if approx_kl.detach().cpu().item() > self.target_kl:
                        early_stop = True

                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()), self.max_grad_norm)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer_critic.step()

                loss_actor_sum += float(loss_actor.detach().cpu().item())
                loss_critic_sum += float(loss_critic.detach().cpu().item())
                entropy_sum += float(entropy.detach().cpu().mean().item())
                clip_frac_sum += float(clip_frac.detach().cpu().item())
                num_minibatches += 1

                if early_stop:
                    break

            if early_stop:
                break

        self.clear_memory()

        if num_minibatches == 0:
            return {}
        return {
            "loss_actor": loss_actor_sum / num_minibatches,
            "loss_critic": loss_critic_sum / num_minibatches,
            "entropy": entropy_sum / num_minibatches,
            "approx_kl": approx_kl_sum / max(num_minibatches, 1),
            "clip_frac": clip_frac_sum / max(num_minibatches, 1),
        }


    def save(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        torch.save(self.log_std, f"{path_prefix}_log_std.pth")

    def load(self, path_prefix, map_location: Optional[str] = None):
        if map_location is None:
            map_location = self.device

        actor_state = torch.load(f"{path_prefix}_actor.pth", map_location=map_location)
        critic_state = torch.load(
            f"{path_prefix}_critic.pth", map_location=map_location
        )
        loaded_log_std = torch.load(f"{path_prefix}_log_std.pth", map_location=map_location)

        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)
        if isinstance(loaded_log_std, torch.Tensor):
            self.log_std.data.copy_(loaded_log_std.to(self.device))
        else:
            # Fallback for unexpected serialization types.
            self.log_std = torch.nn.Parameter(torch.as_tensor(loaded_log_std).to(self.device))

        self.eval()

        return self