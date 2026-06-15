from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from guided_rl.common.interfaces import StepLogger
from guided_rl.common.types import StepData


@dataclass
class LoggerConfig:
    output_dir: str
    episode_name: str = "episode"
    save_jsonl: bool = True
    save_summary: bool = True
    overwrite: bool = False


class Logger(StepLogger):
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._step_path = self.output_dir / f"{config.episode_name}.jsonl"
        self._summary_path = self.output_dir / f"{config.episode_name}_summary.json"
        if config.overwrite:
            if self._step_path.exists():
                self._step_path.unlink()
            if self._summary_path.exists():
                self._summary_path.unlink()
        self._step_index = 0
        self._episode_reward = 0.0
        self._terminated = False
        self._truncated = False

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def log_step(self, step_record: StepData) -> None:
        payload = {
            "obs_features": {
                "ux": step_record.obs_features.ux if step_record.obs_features is not None else None,
                "uy": step_record.obs_features.uy if step_record.obs_features is not None else None,
                "angle_x": step_record.obs_features.angle_x if step_record.obs_features is not None else None,
                "angle_y": step_record.obs_features.angle_y if step_record.obs_features is not None else None,
                "output_x": step_record.obs_features.output_x if step_record.obs_features is not None else None,
                "output_y": step_record.obs_features.output_y if step_record.obs_features is not None else None,
                "depth": step_record.obs_features.depth if step_record.obs_features is not None else None,
                "visible": step_record.obs_features.visible if step_record.obs_features is not None else None,
            },
            "base_action": {
                "ul": step_record.base_action.ul if step_record.base_action is not None else None,
                "ur": step_record.base_action.ur if step_record.base_action is not None else None,
                "dl": step_record.base_action.dl if step_record.base_action is not None else None,
                "dr": step_record.base_action.dr if step_record.base_action is not None else None,
            } if step_record.base_action is not None else None,
            "rl_action": {
                "ul": step_record.rl_action.ul if step_record.rl_action is not None else None,
                "ur": step_record.rl_action.ur if step_record.rl_action is not None else None,
                "dl": step_record.rl_action.dl if step_record.rl_action is not None else None,
                "dr": step_record.rl_action.dr if step_record.rl_action is not None else None,
            } if step_record.rl_action is not None else None,
            "executable_action": {
                "ul": step_record.executable_action.ul if step_record.executable_action is not None else None,
                "ur": step_record.executable_action.ur if step_record.executable_action is not None else None,
                "dl": step_record.executable_action.dl if step_record.executable_action is not None else None,
                "dr": step_record.executable_action.dr if step_record.executable_action is not None else None,
            } if step_record.executable_action is not None else None,
            "env_feedback": {
                "reward": step_record.env_feedback.reward if step_record.env_feedback is not None else None,
                "terminated": step_record.env_feedback.terminated if step_record.env_feedback is not None else None,
                "truncated": step_record.env_feedback.truncated if step_record.env_feedback is not None else None,
                "raw_obs": {
                    "pos": step_record.env_feedback.raw_obs.pos.tolist() if step_record.env_feedback is not None else None,
                    "quat": step_record.env_feedback.raw_obs.quat.tolist() if step_record.env_feedback is not None else None,
                    "euler": step_record.env_feedback.raw_obs.euler.tolist() if step_record.env_feedback is not None else None,
                    "vel": step_record.env_feedback.raw_obs.vel.tolist() if step_record.env_feedback is not None else None,
                    "omega": step_record.env_feedback.raw_obs.omega.tolist() if step_record.env_feedback is not None else None,
                    "force": step_record.env_feedback.raw_obs.force.tolist() if step_record.env_feedback is not None else None,
                    "target_pos": step_record.env_feedback.raw_obs.target_pos.tolist() if step_record.env_feedback is not None else None,
                    "step": step_record.env_feedback.raw_obs.step if step_record.env_feedback is not None else None,
                    "time": step_record.env_feedback.raw_obs.time if step_record.env_feedback is not None else None,
                } if step_record.env_feedback is not None else None,
            } if step_record.env_feedback is not None else None,
        }

        reward = 0.0
        terminated = False
        truncated = False
        if step_record.env_feedback is not None:
            reward = float(step_record.env_feedback.reward)
            terminated = bool(step_record.env_feedback.terminated)
            truncated = bool(step_record.env_feedback.truncated)

        self._step_index += 1
        self._episode_reward += reward
        self._terminated = self._terminated or terminated
        self._truncated = self._truncated or truncated

        if self.config.save_jsonl:
            with self._step_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=self._json_default) + "\n")

        if self.config.save_summary and (terminated or truncated):
            self.flush_summary()

    def flush_summary(self) -> None:
        if not self.config.save_summary:
            return
        summary = {
            "episode_name": self.config.episode_name,
            "steps": self._step_index,
            "episode_reward": float(self._episode_reward),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
        }
        with self._summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


__all__ = ["Logger", "LoggerConfig"]
