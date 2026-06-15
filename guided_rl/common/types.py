from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObservationProcessorConfig:
    fx: float = 1.0
    fy: float = 1.0
    cx: float = 0.0
    cy: float = 0.0
    vis_depth_min: float = 0.1
    vis_hfov_deg: float = 120.0
    vis_vfov_deg: float = 90.0
    depth_proj_min: float = 0.1
    uv_err_clip: float = 20.0


@dataclass
class BaseActionMakerConfig:
    dt: float = 0.01
    fx: float = 1.0
    fy: float = 1.0
    servo_limit: float = 30.0
    mix_matrix: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, 1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [-1.0, -1.0],
            ],
            dtype=np.float32,
        )
    )
    visible_hold_mode: str = "hold_last"
    pitch_adrc: dict = field(
        default_factory=lambda: {
            "r": 10000.0,
            "h": 0.003,
            "w_o": 180.0,
            "b0": 100.0,
            "kp": 1.5,
            "kd": 0.1,
            "alpha_p": 0.75,
            "delta_p": 0.01,
            "alpha_d": 1.0,
            "delta_d": 0.02,
        }
    )
    yaw_adrc: dict = field(
        default_factory=lambda: {
            "r": 10000.0,
            "h": 0.003,
            "w_o": 180.0,
            "b0": 100.0,
            "kp": 1.5,
            "kd": 0.1,
            "alpha_p": 0.75,
            "delta_p": 0.01,
            "alpha_d": 1.0,
            "delta_d": 0.02,
        }
    )


@dataclass
class RLActionMakerConfig:
    use_rl: bool = True
    model_path: str = ""
    algorithm: str = "auto"
    device: str = "cpu"
    deterministic: bool = True
    agent_config: dict = field(default_factory=dict)
    obs_dim: int | None = None
    action_dim: int | None = None
    step_dt: float = 0.01
    update_interval_steps: int = 1


@dataclass
class ActionIntegratorConfig:
    servo_limit_abs: float = 45.0


@dataclass
class LoggerConfig:
    output_dir: str = "./logs"
    episode_name: str = "episode"
    save_jsonl: bool = True
    save_summary: bool = True
    overwrite: bool = False


@dataclass
class DartEnvConfig:
    dt: float = 0.01
    max_steps: int = 2000
    target_pos: np.ndarray = field(default_factory=lambda: np.array([10.0, 0.0, 0.0], dtype=np.float32))
    aero_weight_path: str = ""
    device: str = "cpu"
    dynamics: dict = field(default_factory=dict)
    initial_state: dict = field(default_factory=dict)
    camera: ObservationProcessorConfig = field(default_factory=ObservationProcessorConfig)
    reward: dict = field(default_factory=dict)
    termination: dict = field(default_factory=dict)
    reset_noise: dict = field(default_factory=dict)
    action_clip: float = 30.0


@dataclass
class RewardConfig:
    target_weight: float = 1.0
    distance_weight: float = 1.0
    speed_weight: float = 0.0
    alive_reward: float = 0.0


@dataclass
class TerminationConfig:
    target_distance_threshold: float = 1.0
    min_altitude: float = -100.0
    max_time: float | None = None


@dataclass
class RawEnvObs:
    pos: np.ndarray # 规范世界坐标系是X：右向；Y：前向；Z：向上
    quat: np.ndarray
    euler: np.ndarray
    vel: np.ndarray
    omega: np.ndarray
    force: np.ndarray
    target_pos: np.ndarray
    step: int = 0
    time: float = 0.0


@dataclass
class ObservationFeatures:
    ux: float  # 约定右正
    uy: float  # 约定下正
    angle_x: float
    angle_y: float
    output_x: float
    output_y: float
    depth: float | None = None
    visible: bool = True
    info: dict = field(default_factory=dict)

@dataclass
class DecisionPack1:
    output_x: float
    output_y: float
    pitch: float
    roll: float
    yaw: float
    gyro_pitch: float
    gyro_roll: float
    gyro_yaw: float
    time: float
    visible: bool = True


@dataclass
class BaseAction:
    ul: float
    ur: float
    dl: float
    dr: float


@dataclass
class DecisionPack2:
    output_x: float
    output_y: float
    base_action_ul: float
    base_action_ur: float
    base_action_dl: float
    base_action_dr: float
    quat: np.ndarray
    omega: np.ndarray
    time: float



@dataclass
class RLAction:
    ul: float
    ur: float
    dl: float
    dr: float


@dataclass
class ActionIntegrationInput:
    base_action: BaseAction
    rl_action: RLAction


@dataclass
class ExecutableAction:
    ul: float
    ur: float
    dl: float
    dr: float


@dataclass
class EnvFeedback:
    raw_obs: RawEnvObs
    reward: float
    terminated: bool
    truncated: bool


@dataclass
class StepData:
    raw_obs: RawEnvObs | None = None
    obs_features: ObservationFeatures | None = None
    base_action: BaseAction | None = None
    rl_action: RLAction | None = None
    executable_action: ExecutableAction | None = None
    env_feedback: EnvFeedback | None = None
    decision_pack1: DecisionPack1 | None = None
    decision_pack2: DecisionPack2 | None = None
    action_input: ActionIntegrationInput | None = None 


__all__ = [
    "ActionIntegrationInput",
    "ActionIntegratorConfig",
    "BaseAction",
    "BaseActionMakerConfig",
    "DecisionPack1",
    "DecisionPack2",
    "DartEnvConfig",
    "EnvFeedback",
    "ExecutableAction",
    "ObservationFeatures",
    "ObservationProcessorConfig",
    "RLActionMakerConfig",
    "RawEnvObs",
    "RLAction",
    "LoggerConfig",
    "RewardConfig",
    "StepData",
    "TerminationConfig",
]
