from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow running as: python scripts/train.py or python refactor/scripts/train.py.
# Ensure the refactor package shadows the legacy top-level guided_rl package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
for p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    sys.path.insert(0, p)

from guided_rl.action import ActionIntegrator
from guided_rl.common.enums import DataRequest, DataTag
from guided_rl.common.types import EnvFeedback
from guided_rl.config import ConfigLoader
from guided_rl.decision import BaseActionMaker
from guided_rl.env import DartEnv
from guided_rl.logging import Logger
from guided_rl.observation import ObservationProcessor
from guided_rl.orchestrator import EpisodeRunner
from guided_rl.rl import RLActionMaker
from guided_rl.rl.agent.ppo_agent import PPOAgent
from guided_rl.rl.agent.sac_agent import SACAgent
from guided_rl.runtime import DataHub


@dataclass
class TrainParams:
    project_root: Path
    config_path: Path
    env_config: dict[str, Any]
    train_config: dict[str, Any]
    raw: dict[str, Any]


def load_train_params(script_path: str) -> TrainParams:
    project_root = Path(script_path).resolve().parents[1]
    config_path = project_root / "config" / "config.yaml"
    loader = ConfigLoader()
    raw = loader.load(str(config_path))
    return TrainParams(
        project_root=project_root,
        config_path=config_path,
        env_config=raw.get("env", {}),
        train_config=raw.get("train", {}),
        raw=raw,
    )


def _build_components(params: TrainParams):
    loader = ConfigLoader()
    env_cfg = loader.build_env_config(params.env_config)
    obs_cfg = env_cfg.camera
    base_cfg = loader.build_base_action_config(params.env_config.get("base_action_maker", {}))
    rl_cfg = loader.build_rl_action_config(params.env_config.get("rl_action_maker", {}))
    action_cfg = loader.build_action_integrator_config(params.env_config.get("action_integrator", {}))
    logger_cfg = loader.build_logger_config(params.train_config.get("logging", params.raw.get("eval", {}).get("logging", {})))

    env = DartEnv(env_cfg)
    data_hub = DataHub(history_len=int(params.env_config.get("history_len", 3)))
    obs_processor = ObservationProcessor(obs_cfg)
    base_action_maker = BaseActionMaker(base_cfg)
    rl_action_maker = RLActionMaker(rl_cfg)
    action_integrator = ActionIntegrator(action_cfg)
    logger = Logger(logger_cfg)

    runner = EpisodeRunner(
        env=env,
        data_hub=data_hub,
        observation_processor=obs_processor,
        base_action_maker=base_action_maker,
        rl_action_maker=rl_action_maker,
        action_integrator=action_integrator,
        logger=logger,
    )
    return env, runner, logger, env_cfg, rl_cfg


def _make_agent(algorithm: str, obs_dim: int, action_dim: int, device: str, agent_config: dict[str, Any]):
    cfg = dict(agent_config or {})
    if algorithm == "sac":
        return SACAgent(obs_dim, action_dim, device=device, **cfg)
    return PPOAgent(obs_dim, action_dim, device=device, **cfg)


def _print_training_summary(history: list[dict[str, Any]]) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    cols = [c for c in ["episode", "reward", "steps", "loss_actor", "loss_critic", "entropy", "approx_kl", "clip_frac", "loss_q", "alpha"] if c in df.columns]
    print("Training summary:")
    print(df[cols].tail(10).to_string(index=False))


def _save_snapshot(params: TrainParams, env_cfg: Any) -> None:
    snapshot_dir = params.project_root / "train_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    with (snapshot_dir / "env.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(env_cfg), f, ensure_ascii=False, indent=2, default=_default)
    with (snapshot_dir / "train.json").open("w", encoding="utf-8") as f:
        json.dump(params.train_config, f, ensure_ascii=False, indent=2, default=_default)


def _scale_rl_action(action: np.ndarray, limit: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size == 0:
        return action
    scaled = action * float(limit)
    return np.clip(scaled, -float(limit), float(limit))


def _policy_action_to_rl_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size == 0:
        return action
    return np.clip(action, -1.0, 1.0)


def main() -> None:
    params = load_train_params(__file__)
    env, runner, logger, env_cfg, rl_cfg = _build_components(params)

    train_cfg = params.train_config
    algorithm = str(train_cfg.get("algorithm", rl_cfg.algorithm or "ppo")).lower()
    if algorithm == "auto":
        algorithm = "sac" if str(rl_cfg.model_path).strip().endswith("q1") else "ppo"
    device = str(train_cfg.get("device", rl_cfg.device or "cpu"))
    episodes = int(train_cfg.get("episodes", 10))
    max_steps_per_episode = int(train_cfg.get("max_steps_per_episode", env_cfg.max_steps))
    save_every = int(train_cfg.get("save_every_episodes", 10))
    save_dir = Path(train_cfg.get("save_dir", params.project_root / "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = train_cfg.get("save_prefix", "policy")
    overwrite = bool(train_cfg.get("overwrite_logs", True))
    if overwrite:
        logger.config.overwrite = True

    if bool(train_cfg.get("save_snapshot", True)):
        _save_snapshot(params, env_cfg)

    obs_dim = int(getattr(rl_cfg, "obs_dim", 0) or 0)
    if obs_dim <= 0:
        obs_dim = 14
    action_dim = int(train_cfg.get("action_dim", 4))
    agent = _make_agent(algorithm, obs_dim, action_dim, device=device, agent_config=dict(train_cfg.get("agent_config", {})))

    # RL action output range. Prefer config, otherwise default to [-30, 30].
    rl_action_limit = train_cfg.get("rl_action_limit", getattr(rl_cfg, "action_limit", None))
    if rl_action_limit is None:
        rl_action_limit = 30.0
    rl_action_limit = float(rl_action_limit)
    if rl_action_limit <= 0:
        rl_action_limit = 30.0

    target_x_min = float(train_cfg.get("target_x_min", -0.5))
    target_x_max = float(train_cfg.get("target_x_max", 0.5))
    target_y = float(train_cfg.get("target_y", 18.0))
    target_z = float(train_cfg.get("target_z", 0.0))
    rng = np.random.default_rng(int(train_cfg.get("seed", 0)))

    history: list[dict[str, Any]] = []
    for episode in range(1, episodes + 1):
        sampled_target = np.asarray([
            rng.uniform(target_x_min, target_x_max),
            target_y,
            target_z,
        ], dtype=np.float32)
        env.target_pos = sampled_target.copy()
        env.config.target_pos = sampled_target.copy()

        raw_obs = env.reset()
        raw_obs.target_pos = sampled_target.copy()
        episode_reward = 0.0
        done = False
        truncated = False
        step_in_ep = 0
        last_obs = None

        while not done and not truncated and step_in_ep < max_steps_per_episode:
            runner.data_hub.begin_step()
            runner.data_hub.input(DataTag.RAW_OBS, raw_obs)
            obs_input = runner.data_hub.output(DataRequest.OBSERVATION_INPUT)
            obs_features = runner.observation_processor.compute(obs_input)
            runner.data_hub.input(DataTag.OBS_FEATURES, obs_features)

            decision_pack1 = runner.data_hub.output(DataRequest.DECISION_PACK1)
            base_action = runner.base_action_maker.compute(decision_pack1)
            runner.data_hub.input(DataTag.BASE_ACTION, base_action)

            decision_pack2 = runner.data_hub.output(DataRequest.DECISION_PACK2)
            obs = runner.rl_action_maker._pack_to_obs(decision_pack2)
            action_np, log_prob, value = agent.act(obs, deterministic=False)
            policy_action_arr = _policy_action_to_rl_action(action_np)
            if policy_action_arr.shape[0] < 4:
                raise ValueError(f"Agent returned invalid action shape: {policy_action_arr.shape}")
            env_rl_action_arr = _scale_rl_action(policy_action_arr[:4], rl_action_limit)

            runner.data_hub.input(DataTag.RL_ACTION, runner.rl_action_maker._to_rl_action(env_rl_action_arr))
            action_input = runner.data_hub.output(DataRequest.ACTION_INTEGRATION_INPUT)
            executable_action = runner.action_integrator.integrate(action_input)
            runner.data_hub.input(DataTag.EXECUTABLE_ACTION, executable_action)

            env_action = np.asarray([executable_action.ul, executable_action.ur, executable_action.dl, executable_action.dr], dtype=np.float32)
            next_raw_obs, reward, done, truncated = env.step(env_action)
            runner.data_hub.input(DataTag.ENV_FEEDBACK, EnvFeedback(raw_obs=next_raw_obs, reward=float(reward), terminated=bool(done), truncated=bool(truncated)))

            if algorithm == "ppo":
                agent.store_transition(
                    obs=obs,
                    action=policy_action_arr[:4],
                    reward=float(reward),
                    done=bool(done or truncated),
                    log_prob=float(log_prob),
                    value=float(value),
                )

            step_record = runner.data_hub.output(DataRequest.STEP_RECORD)
            if logger is not None:
                logger.log_step(step_record)
            runner.data_hub.commit_step()

            raw_obs = next_raw_obs
            last_obs = obs
            episode_reward += float(reward)
            step_in_ep += 1

        if algorithm == "ppo":
            terminal_value = 0.0 if done and not truncated else float(agent.act(last_obs if last_obs is not None else obs, deterministic=True)[2])
            metrics = agent.update(last_value=terminal_value)
        else:
            metrics = {}

        logger.flush_summary()
        history.append({
            "episode": episode,
            "reward": episode_reward,
            "steps": step_in_ep,
            "target_x": float(sampled_target[0]),
            "target_y": float(sampled_target[1]),
            "target_z": float(sampled_target[2]),
            **metrics,
        })
        print(f"Episode {episode:04d} | target={sampled_target.tolist()} | reward={episode_reward:.3f} | steps={step_in_ep} | metrics={metrics}")

        if episode % save_every == 0:
            agent.save(str(save_dir / f"{save_prefix}_{episode:04d}"))

    agent.save(str(save_dir / save_prefix))
    env.close()
    _print_training_summary(history)
    print(f"Training finished. Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
