from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

from guided_rl.common.types import (
    DartEnvConfig,
    EnvFeedback,
    ObservationFeatures,
    ObservationProcessorConfig,
    RawEnvObs,
    RewardConfig,
    TerminationConfig,
)
from guided_rl.env.physics.aerodynamics import AeroModel
from guided_rl.env.physics.dynamics import DartDynamics, quat_to_euler


class RewardCalculator:
    def __init__(self, config: RewardConfig | dict | None = None, termination_config: TerminationConfig | dict | None = None):
        if config is None:
            config = RewardConfig()
        if isinstance(config, dict):
            config = RewardConfig(**config)
        if termination_config is None:
            termination_config = TerminationConfig()
        if isinstance(termination_config, dict):
            termination_config = TerminationConfig(**termination_config)
        self.config = config
        self.termination_config = termination_config
        self._parabola_cache: dict[str, float] | None = None

    def reset(self, initial_state: dict, target_pos: np.ndarray) -> None:
        pos = np.asarray(initial_state.get("pos"), dtype=np.float32).reshape(-1)
        vel = np.asarray(initial_state.get("vel"), dtype=np.float32).reshape(-1)
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(-1)

        if pos.shape[0] != 3 or vel.shape[0] != 3 or target_pos.shape[0] != 3:
            raise ValueError("pos, vel and target_pos must have shape (3,)")

        y0 = float(pos[1])
        z0 = float(pos[2])
        vy0 = float(vel[1])
        vz0 = float(vel[2])
        y_target = float(target_pos[1])
        z_target = float(target_pos[2])

        if abs(y_target - y0) <= 1e-6:
            self._parabola_cache = {"y0": y0, "z0": z0, "a": 0.0, "b": 0.0 if abs(vy0) <= 1e-6 else vz0 / vy0}
            return

        b = 0.0 if abs(vy0) <= 1e-6 else vz0 / vy0
        dy_target = y_target - y0
        a = (z_target - z0 - b * dy_target) / (dy_target * dy_target)
        self._parabola_cache = {"y0": y0, "z0": z0, "a": float(a), "b": float(b)}

    def _parabola_height(self, y: float) -> float:
        if self._parabola_cache is None:
            return 0.0
        dy = y - self._parabola_cache["y0"]
        return float(
            self._parabola_cache["z0"]
            + self._parabola_cache["b"] * dy
            + self._parabola_cache["a"] * dy * dy
        )

    def compute(self, state: dict, target_pos: np.ndarray,terminated: bool) -> float:
        pos = np.asarray(state.get("pos"), dtype=np.float32).reshape(-1)
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(-1)

        if pos.shape[0] != 3 or target_pos.shape[0] != 3:
            raise ValueError("pos and target_pos must have shape (3,)")
        if self._parabola_cache is None:
            raise RuntimeError("RewardCalculator.reset must be called before compute")

        # 横向偏差：按你定义的 (pos[0] - target_pos[0]) * pos[1]
        lateral_error = abs(float(pos[0] - target_pos[0]) )

        # 竖直偏差：用初始位置、初始速度、目标位置拟合的固定抛物线来评估
        parabola_z = self._parabola_height(float(pos[1]))
        vertical_error = abs(float(pos[2] - parabola_z))

        dist = float(np.linalg.norm(pos - target_pos))

        # if terminated:
        #     if dist <= self.termination_config.target_distance_threshold :
        #         hit_bonus = 300.0
        #         print("hit !!!!!!!!!!!!!!!!!!!!\n")
        #     else:
        #         hit_bonus = -dist*1000.0
        # else:
        #     hit_bonus = 0.0


        if dist <= self.termination_config.target_distance_threshold :
            hit_bonus = 300.0
            print("hit !!!!!!!!!!!!!!!!!!!!\n")
        else:
            hit_bonus = 0.0
        #ßprint(f"lateral_error: {lateral_error}, vertical_error: {vertical_error}, hit_bonus: {hit_bonus}")

        return float(
            self.config.alive_reward
             - self.config.distance_weight * (lateral_error + vertical_error*100.0)
            + hit_bonus
        )


class TerminationChecker:
    def __init__(self, config: TerminationConfig | dict | None = None):
        if config is None:
            config = TerminationConfig()
        if isinstance(config, dict):
            config = TerminationConfig(**config)
        self.config = config

    def check(self, state: dict, target_pos: np.ndarray, step_count: int, max_steps: int) -> tuple[bool, bool]:
        pos = np.asarray(state.get("pos"), dtype=np.float32)
        dist = float(np.linalg.norm(pos - target_pos))
        z_reached_ground = pos[2] <= 0.0
        terminated = (
            dist <= self.config.target_distance_threshold
            or z_reached_ground
        )
        if dist <= self.config.target_distance_threshold:
            print("hit")
        truncated = step_count >= max_steps if self.config.max_time is None else False
        return bool(terminated), bool(truncated)


class DartEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, config: DartEnvConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            camera_cfg = config.get("camera", {})
            if not isinstance(camera_cfg, ObservationProcessorConfig):
                camera_cfg = ObservationProcessorConfig(**camera_cfg)
            config = DartEnvConfig(
                dt=float(config.get("dt", 0.01)),
                max_steps=int(config.get("max_steps", 2000)),
                target_pos=np.asarray(config.get("target_pos", [10.0, 0.0, 0.0]), dtype=np.float32),
                aero_weight_path=str(config.get("aero_weight_path", "")),
                device=str(config.get("device", "cpu")),
                dynamics=dict(config.get("dynamics", {})),
                initial_state=dict(config.get("initial_state", {})),
                camera=camera_cfg,
                reward=dict(config.get("reward", {})),
                termination=dict(config.get("termination", {})),
                reset_noise=dict(config.get("reset_noise", {})),
                action_clip=float(config.get("action_clip", 30.0)),
            )
        self.config = config

        self.dt = float(self.config.dt)
        self.max_steps = int(self.config.max_steps)
        self.target_pos = np.asarray(self.config.target_pos, dtype=np.float32)
        self.aero = AeroModel(self.config.aero_weight_path, self.config.device)
        self.dynamics = DartDynamics(self.aero, {
            "dynamics": self.config.dynamics,
            "initial_state": self.config.initial_state,
        })
        self.reset_noise_cfg = dict(self.config.reset_noise)
        self.reward_calculator = RewardCalculator(self.config.reward, self.config.termination)
        self.termination_checker = TerminationChecker(self.config.termination)
        self.camera_cfg = self.config.camera
        self.action_clip = float(self.config.action_clip)

        self.action_space = spaces.Box(low=-self.action_clip, high=self.action_clip, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self.step_count = 0
        self.episode_time = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.step_count = 0
        self.episode_time = 0.0
        self.dynamics = DartDynamics(self.aero, {
            "dynamics": self.config.dynamics,
            "initial_state": self.config.initial_state,
        })
        self.reward_calculator.reset({"pos": self.dynamics.pos, "vel": self.dynamics.vel}, self.target_pos)

        pos_scale = float(self.reset_noise_cfg.get("pos_scale", 0.0))
        vel_scale = float(self.reset_noise_cfg.get("vel_scale", 0.0))
        omega_scale = float(self.reset_noise_cfg.get("omega_scale", 0.0))
        if pos_scale > 0:
            self.dynamics.pos += self.np_random.normal(0.0, pos_scale, size=3).astype(np.float32)
        if vel_scale > 0:
            self.dynamics.vel += self.np_random.normal(0.0, vel_scale, size=3).astype(np.float32)
        if omega_scale > 0:
            self.dynamics.omega += self.np_random.normal(0.0, omega_scale, size=3).astype(np.float32)

        raw_obs = self._build_raw_obs(self.dynamics)
        return raw_obs

    def step(self, action: np.ndarray):
        servo_cmd = np.asarray(action, dtype=np.float32).reshape(-1)
        if servo_cmd.shape != (4,):
            raise ValueError(f"DartEnv.step expects servo command shape (4,), got {servo_cmd.shape}")
        servo_cmd = np.clip(servo_cmd, -self.action_clip, self.action_clip)

        self.step_count += 1
        self.episode_time += self.dt

        state = self.dynamics.step(servo_cmd, self.dt)
        terminated, truncated = self.termination_checker.check(state, self.target_pos, self.step_count, self.max_steps)
        reward = self.reward_calculator.compute(state, self.target_pos,terminated)
        if self.step_count >= self.max_steps:
            truncated = True

        raw_obs = self._build_raw_obs(state)
        return raw_obs, reward, terminated, truncated

    def _build_raw_obs(self, state: dict | DartDynamics) -> RawEnvObs:
        if isinstance(state, DartDynamics):
            quat = np.asarray(state.q, dtype=np.float32).copy()
            state = {
                "pos": state.pos,
                "quat": quat,
                "euler": quat_to_euler(quat),
                "vel": state.vel,
                "omega": state.omega,
                "force": np.zeros(3, dtype=np.float32),
            }

        return RawEnvObs(
            pos=np.asarray(state["pos"], dtype=np.float32).copy(),
            quat=np.asarray(state["quat"], dtype=np.float32).copy(),
            euler=np.asarray(state["euler"], dtype=np.float32).copy(),
            vel=np.asarray(state["vel"], dtype=np.float32).copy(),
            omega=np.asarray(state["omega"], dtype=np.float32).copy(),
            force=np.asarray(state["force"], dtype=np.float32).copy(),
            target_pos=self.target_pos.copy(),
            step=self.step_count,
            time=self.episode_time,
        )


__all__ = ["DartEnv", "RewardCalculator", "TerminationChecker"]
