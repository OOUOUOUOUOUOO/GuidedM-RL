from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow running as: python scripts/eval_test.py or python refactor/scripts/eval_test.py.
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
from guided_rl.common.types import EnvFeedback, RLAction
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
class EvalParams:
    project_root: Path
    config_path: Path
    env_config: dict[str, Any]
    eval_test_config: dict[str, Any]
    raw: dict[str, Any]


def load_eval_params(script_path: str) -> EvalParams:
    project_root = Path(script_path).resolve().parents[1]
    config_path = project_root / "config" / "config.yaml"
    loader = ConfigLoader()
    raw = loader.load(str(config_path))
    return EvalParams(
        project_root=project_root,
        config_path=config_path,
        env_config=raw.get("env", {}),
        eval_test_config=raw.get("eval_test", {}),
        raw=raw,
    )


def _build_components(params: EvalParams):
    loader = ConfigLoader()
    env_cfg = loader.build_env_config(params.env_config)
    obs_cfg = env_cfg.camera
    base_cfg = loader.build_base_action_config(params.env_config.get("base_action_maker", {}))
    rl_cfg = loader.build_rl_action_config(params.env_config.get("rl_action_maker", {}))
    action_cfg = loader.build_action_integrator_config(params.env_config.get("action_integrator", {}))
    logger_cfg = loader.build_logger_config(params.raw.get("eval", {}).get("logging", {}))

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


def _print_eval_summary(history: list[dict[str, Any]]) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    cols = [c for c in ["episode", "reward", "steps", "hit"] if c in df.columns]
    print("Evaluation summary:")
    print(df[cols].tail(10).to_string(index=False))


def _save_snapshot(params: EvalParams, env_cfg: Any) -> None:
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
    with (snapshot_dir / "eval_test.json").open("w", encoding="utf-8") as f:
        json.dump(params.eval_test_config, f, ensure_ascii=False, indent=2, default=_default)


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
    params = load_eval_params(__file__)
    env, runner, logger, env_cfg, rl_cfg = _build_components(params)

    eval_cfg = params.eval_test_config if isinstance(params.eval_test_config, dict) else {}
    rl_cfg_dict = params.env_config.get("rl_action_maker", {}) if isinstance(params.env_config.get("rl_action_maker", {}), dict) else {}

    episodes = int(eval_cfg.get("episodes", 100))
    max_steps_per_episode = int(eval_cfg.get("max_steps_per_episode", env_cfg.max_steps))
    seed = int(eval_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    use_rl = bool(rl_cfg_dict.get("use_rl", rl_cfg.use_rl))
    algorithm = str(rl_cfg_dict.get("algorithm", rl_cfg.algorithm or "ppo")).lower()
    if algorithm == "auto":
        algorithm = "sac" if str(rl_cfg.model_path).strip().endswith("q1") else "ppo"
    device = str(rl_cfg_dict.get("device", rl_cfg.device or "cpu"))

    obs_dim = int(getattr(rl_cfg, "obs_dim", 0) or 0)
    if obs_dim <= 0:
        obs_dim = 14
    action_dim = int(getattr(rl_cfg, "action_dim", 0) or 0)
    if action_dim <= 0:
        action_dim = 4

    agent = None
    if use_rl:
        model_path = str(rl_cfg_dict.get("model_path", rl_cfg.model_path or "")).strip()
        if not model_path:
            raise ValueError("eval_test requires env.rl_action_maker.model_path when use_rl is true")
        agent = _make_agent(
            algorithm,
            obs_dim,
            action_dim,
            device=device,
            agent_config=dict(rl_cfg_dict.get("agent_config", rl_cfg.agent_config or {})),
        )
        agent.load(model_path)

    rl_action_limit = float(rl_cfg_dict.get("rl_action_limit", 30.0)) if "rl_action_limit" in rl_cfg_dict else 30.0
    if rl_action_limit <= 0:
        rl_action_limit = 30.0

    target_x_min = float(eval_cfg.get("target_x_min", params.raw.get("train", {}).get("target_x_min", -0.5)))
    target_x_max = float(eval_cfg.get("target_x_max", params.raw.get("train", {}).get("target_x_max", 0.5)))
    target_y = float(eval_cfg.get("target_y", params.raw.get("train", {}).get("target_y", 18.0)))
    target_z = float(eval_cfg.get("target_z", params.raw.get("train", {}).get("target_z", 0.0)))

    history: list[dict[str, Any]] = []
    total_hits = 0

    if bool(eval_cfg.get("save_snapshot", True)):
        _save_snapshot(params, env_cfg)

    try:
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
            hit = False

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

                if use_rl and agent is not None:
                    action_np, _, _ = agent.act(obs, deterministic=bool(rl_cfg_dict.get("deterministic", rl_cfg.deterministic)))
                    policy_action_arr = _policy_action_to_rl_action(action_np)
                    if policy_action_arr.shape[0] < 4:
                        raise ValueError(f"Agent returned invalid action shape: {policy_action_arr.shape}")
                    env_rl_action_arr = _scale_rl_action(policy_action_arr[:4], rl_action_limit)
                    runner.data_hub.input(DataTag.RL_ACTION, runner.rl_action_maker._to_rl_action(env_rl_action_arr))
                else:
                    runner.data_hub.input(DataTag.RL_ACTION, RLAction(ul=0.0, ur=0.0, dl=0.0, dr=0.0))

                action_input = runner.data_hub.output(DataRequest.ACTION_INTEGRATION_INPUT)
                executable_action = runner.action_integrator.integrate(action_input)
                runner.data_hub.input(DataTag.EXECUTABLE_ACTION, executable_action)

                env_action = np.asarray([executable_action.ul, executable_action.ur, executable_action.dl, executable_action.dr], dtype=np.float32)
                next_raw_obs, reward, done, truncated = env.step(env_action)
                runner.data_hub.input(DataTag.ENV_FEEDBACK, EnvFeedback(raw_obs=next_raw_obs, reward=float(reward), terminated=bool(done), truncated=bool(truncated)))

                step_record = runner.data_hub.output(DataRequest.STEP_RECORD)
                if logger is not None:
                    logger.log_step(step_record)
                runner.data_hub.commit_step()

                raw_obs = next_raw_obs
                episode_reward += float(reward)
                step_in_ep += 1

            pos = np.asarray(raw_obs.pos, dtype=np.float32).reshape(-1)
            dist = float(np.linalg.norm(pos - sampled_target))
            if bool(done) and dist <= float(0.15):
                hit = True
                total_hits += 1

            history.append({
                "episode": episode,
                "reward": episode_reward,
                "steps": step_in_ep,
                "target_x": float(sampled_target[0]),
                "target_y": float(sampled_target[1]),
                "target_z": float(sampled_target[2]),
                "hit": hit,
            })
            print(f"Episode {episode:04d} | target={sampled_target.tolist()} | reward={episode_reward:.3f} | steps={step_in_ep} | hit={hit}")

        logger.flush_summary()
        _print_eval_summary(history)
        total_episodes = len(history)
        hit_rate = (total_hits / total_episodes) if total_episodes else 0.0
        print("Evaluation test summary:")
        print(f"episodes: {total_episodes}")
        print(f"hits: {total_hits}")
        print(f"hit_rate: {hit_rate:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
