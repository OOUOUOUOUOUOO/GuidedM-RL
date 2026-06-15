from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from guided_rl.common.types import (
    ActionIntegratorConfig,
    BaseActionMakerConfig,
    DartEnvConfig,
    ObservationProcessorConfig,
    RLActionMakerConfig,
    LoggerConfig,
    RewardConfig,
    TerminationConfig,
)


class ConfigLoader:
    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
            return None
        if isinstance(value, dict):
            return {k: ConfigLoader._normalize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ConfigLoader._normalize_value(v) for v in value]
        return value

    def load(self, path: str) -> dict[str, Any]:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            return self._normalize_value(yaml.safe_load(f) or {})

    @staticmethod
    def build_dataclass(cls, data: dict[str, Any]):
        if data is None:
            data = {}
        return cls(**data)

    def build_env_config(self, data: dict[str, Any]) -> DartEnvConfig:
        camera = data.get("camera", {})
        if not isinstance(camera, ObservationProcessorConfig):
            camera = self.build_dataclass(ObservationProcessorConfig, camera)
        return DartEnvConfig(
            dt=float(data.get("dt", 0.01)),
            max_steps=int(data.get("max_steps", 2000)),
            target_pos=data.get("target_pos", None) if data.get("target_pos", None) is not None else DartEnvConfig().target_pos,
            aero_weight_path=str(data.get("aero_weight_path", "")),
            device=str(data.get("device", "cpu")),
            dynamics=dict(data.get("dynamics", {})),
            initial_state=dict(data.get("initial_state", {})),
            camera=camera,
            reward=dict(data.get("reward", {})),
            termination=dict(data.get("termination", {})),
            reset_noise=dict(data.get("reset_noise", {})),
            action_clip=float(data.get("action_clip", 30.0)),
        )

    def build_base_action_config(self, data: dict[str, Any]) -> BaseActionMakerConfig:
        return BaseActionMakerConfig(
            dt=float(data.get("dt", 0.01)),
            fx=float(data.get("fx", 1.0)),
            fy=float(data.get("fy", 1.0)),
            servo_limit=float(data.get("servo_limit", 30.0)),
            mix_matrix=data.get("mix_matrix", None) if data.get("mix_matrix", None) is not None else BaseActionMakerConfig().mix_matrix,
            visible_hold_mode=str(data.get("visible_hold_mode", "hold_last")),
            pitch_adrc=dict(data.get("pitch_adrc", BaseActionMakerConfig().pitch_adrc)),
            yaw_adrc=dict(data.get("yaw_adrc", BaseActionMakerConfig().yaw_adrc)),
        )

    def build_action_integrator_config(self, data: dict[str, Any]) -> ActionIntegratorConfig:
        return ActionIntegratorConfig(servo_limit_abs=float(data.get("servo_limit_abs", 45.0)))

    def build_rl_action_config(self, data: dict[str, Any]) -> RLActionMakerConfig:
        return RLActionMakerConfig(
            use_rl=bool(data.get("use_rl", True)),
            model_path=str(data.get("model_path", "")),
            algorithm=str(data.get("algorithm", "auto")),
            device=str(data.get("device", "cpu")),
            deterministic=bool(data.get("deterministic", True)),
            agent_config=dict(data.get("agent_config", {})),
            obs_dim=data.get("obs_dim", None),
            action_dim=data.get("action_dim", None),
            step_dt=float(data.get("step_dt", 0.01)),
            update_interval_steps=int(data.get("update_interval_steps", 1)),
        )

    def build_logger_config(self, data: dict[str, Any]) -> LoggerConfig:
        return LoggerConfig(
            output_dir=str(data.get("output_dir", "./logs")),
            episode_name=str(data.get("episode_name", "episode")),
            save_jsonl=bool(data.get("save_jsonl", True)),
            save_summary=bool(data.get("save_summary", True)),
            overwrite=bool(data.get("overwrite", False)),
        )

    def build_reward_config(self, data: dict[str, Any]) -> RewardConfig:
        return RewardConfig(**data)

    def build_termination_config(self, data: dict[str, Any]) -> TerminationConfig:
        return TerminationConfig(**data)


def load_eval_params(script_path: str):
    project_root = Path(script_path).resolve().parents[1]
    config_path = project_root / "config" / "config.yaml"
    loader = ConfigLoader()
    raw = loader.load(str(config_path))
    return SimpleNamespace(
        project_root=project_root,
        config_path=config_path,
        env_config=raw.get("env", {}),
        eval_config=raw.get("eval", {}),
        raw=raw,
    )
