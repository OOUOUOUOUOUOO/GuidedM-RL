from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import torch

from guided_rl.common.types import DecisionPack2, RLAction, RLActionMakerConfig
from guided_rl.rl.agent.ppo_agent import PPOAgent
from guided_rl.rl.agent.sac_agent import SACAgent


@dataclass
class _RLPolicyState:
    algorithm: str
    obs_dim: int
    action_dim: int
    agent: Any


class RLActionMaker:
    def __init__(self, config: RLActionMakerConfig):
        self.config = config
        self._state: _RLPolicyState | None = None
        self._last_action: RLAction | None = None
        self._last_update_step: int | None = None

    def _detect_algorithm(self) -> str:
        if self.config.algorithm and self.config.algorithm.strip().lower() != "auto":
            return self.config.algorithm.strip().lower()
        if self.config.model_path:
            q1_path = f"{self.config.model_path}_q1.pth"
            try:
                import os

                if os.path.exists(q1_path):
                    return "sac"
            except Exception:
                pass
        return "ppo"

    def _infer_action_dim(self, model_path: str, algorithm: str, map_location: str = "cpu") -> int | None:
        import os

        actor_path = f"{model_path}_actor.pth"
        if not os.path.exists(actor_path):
            return None

        state = torch.load(actor_path, map_location=map_location)
        if not isinstance(state, dict):
            return None

        if algorithm == "sac":
            for key, value in state.items():
                if key.endswith("mean_head.weight") and getattr(value, "ndim", 0) == 2:
                    return int(value.shape[0])
                if key.endswith("mean_head.bias") and getattr(value, "ndim", 0) == 1:
                    return int(value.shape[0])

        actor_path = f"{model_path}_actor.pth"
        if not os.path.exists(actor_path):
            return None
        state = torch.load(actor_path, map_location=map_location)
        if not isinstance(state, dict):
            return None
        if "4.weight" in state:
            return int(state["4.weight"].shape[0])
        linear_weights = [v for k, v in state.items() if k.endswith(".weight") and getattr(v, "ndim", 0) == 2]
        if not linear_weights:
            return None
        head = sorted(linear_weights, key=lambda t: t.shape[1])[0]
        return int(head.shape[0])

    def _build_agent(self, algorithm: str, obs_dim: int, action_dim: int):
        cfg = dict(self.config.agent_config)
        device = self.config.device
        if algorithm == "sac":
            return SACAgent(obs_dim, action_dim, device=device, **cfg)
        return PPOAgent(obs_dim, action_dim, device=device, **cfg)

    def _ensure_ready(self, pack: DecisionPack2) -> None:
        if self._state is not None:
            return

        obs = self._pack_to_obs(pack)
        obs_dim = int(self.config.obs_dim or obs.shape[0])
        algorithm = self._detect_algorithm()

        action_dim = self.config.action_dim
        if action_dim is None:
            inferred = self._infer_action_dim(self.config.model_path, algorithm, map_location=self.config.device)
            action_dim = inferred if inferred is not None else 4

        if not self.config.model_path:
            raise ValueError("RLActionMakerConfig.model_path is required")

        agent = self._build_agent(algorithm, obs_dim, int(action_dim))
        agent.load(self.config.model_path)

        self._state = _RLPolicyState(
            algorithm=algorithm,
            obs_dim=obs_dim,
            action_dim=int(action_dim),
            agent=agent,
        )
        self._last_action = None
        self._last_update_step = None

    def _pack_to_obs(self, pack: DecisionPack2) -> np.ndarray:
        return np.array(
            [
                float(pack.output_x*10),
                float(pack.output_y*10),
                float(pack.base_action_ul/10),
                float(pack.base_action_ur/10),
                float(pack.base_action_dl/10),
                float(pack.base_action_dr/10),
                *np.asarray(pack.quat, dtype=np.float32).reshape(-1).tolist(),
                *np.asarray(pack.omega, dtype=np.float32).reshape(-1).tolist(),
                float(pack.time),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _to_rl_action(action: np.ndarray) -> RLAction:
        if action.shape[0] < 4:
            raise ValueError(f"RL agent returned action with invalid shape: {action.shape}")
        return RLAction(
            ul=float(action[0]),
            ur=float(action[1]),
            dl=float(action[2]),
            dr=float(action[3]),
        )

    def compute(self, pack: DecisionPack2) -> RLAction:
        if not self.config.use_rl:
            return RLAction(ul=0.0, ur=0.0, dl=0.0, dr=0.0)
        self._ensure_ready(pack)
        assert self._state is not None

        current_step = int(round(float(pack.time) / max(float(self.config.step_dt), 1e-6)))
        interval = max(int(self.config.update_interval_steps), 1)
        if self._last_action is not None and self._last_update_step is not None:
            if current_step - self._last_update_step < interval:
                return self._last_action

        obs = self._pack_to_obs(pack)
        action, _, _ = self._state.agent.act(obs, deterministic=self.config.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        rl_action = self._to_rl_action(action)
        self._last_action = rl_action
        self._last_update_step = current_step
        return rl_action
