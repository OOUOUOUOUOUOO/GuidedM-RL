from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R

# Allow running as: python scripts/eval.py or python refactor/scripts/eval.py.
# Ensure the refactor package shadows the legacy top-level guided_rl package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
for p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    sys.path.insert(0, p)

from guided_rl.action import ActionIntegrator
from guided_rl.config import ConfigLoader, load_eval_params
from guided_rl.decision import BaseActionMaker
from guided_rl.env import DartEnv
from guided_rl.logging import Logger
from guided_rl.observation import ObservationProcessor
from guided_rl.orchestrator import EpisodeRunner
from guided_rl.rl import RLActionMaker
from guided_rl.runtime import DataHub


def _build_components(params):
    loader = ConfigLoader()
    env_cfg = loader.build_env_config(params.env_config)
    obs_cfg = env_cfg.camera
    base_cfg = loader.build_base_action_config(params.env_config.get("base_action_maker", {}))
    rl_cfg = loader.build_rl_action_config(params.env_config.get("rl_action_maker", {}))
    action_cfg = loader.build_action_integrator_config(params.env_config.get("action_integrator", {}))
    logger_cfg = loader.build_logger_config(params.eval_config.get("logging", {}))

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
    return env, runner, logger, env_cfg


def _load_jsonl_log(log_file: str) -> pd.DataFrame:
    rows = []
    with Path(log_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Log file is empty: {log_file}")
    return df


def _extract_series(df: pd.DataFrame, *path: str, default=np.nan):
    cur = df
    for key in path:
        if isinstance(cur, pd.DataFrame):
            if key not in cur.columns:
                return np.full(len(df), default, dtype=float)
            cur = cur[key]
        else:
            cur = cur.apply(lambda x: x.get(key, default) if isinstance(x, dict) else default)

    if isinstance(cur, pd.Series) and cur.apply(lambda x: isinstance(x, dict)).any():
        dict_rows = cur.apply(lambda x: x if isinstance(x, dict) else {})
        keys = list(dict.fromkeys(key for row in dict_rows for key in row))
        if not keys:
            return np.full(len(df), default, dtype=float)
        values = np.asarray(
            [[row.get(key, default) for key in keys] for row in dict_rows],
            dtype=float,
        )
        return values, keys

    if isinstance(cur, pd.Series) and cur.apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
        seq_rows = cur.apply(lambda x: x if isinstance(x, (list, tuple, np.ndarray)) else [])
        max_len = max((len(row) for row in seq_rows), default=0)
        if max_len == 0:
            return np.full(len(df), default, dtype=float)
        return np.asarray(
            [list(row) + [default] * (max_len - len(row)) for row in seq_rows],
            dtype=float,
        )

    return np.asarray(cur, dtype=float)


def _plot_log(log_file: str) -> None:
    df = _load_jsonl_log(log_file)
    n_frames = len(df)

    times = np.arange(n_frames, dtype=float)
    if "env_feedback" in df.columns:
        raw_time = df["env_feedback"].apply(lambda x: x.get("raw_obs", {}).get("time") if isinstance(x, dict) else None)
        raw_time = pd.to_numeric(raw_time, errors="coerce")
        if raw_time.notna().any():
            times = raw_time.to_numpy(dtype=float)

    ux = _extract_series(df, "obs_features", "ux")
    uy = _extract_series(df, "obs_features", "uy")
    output_x = _extract_series(df, "obs_features", "output_x")
    output_y = _extract_series(df, "obs_features", "output_y")
    reward = np.full(n_frames, np.nan, dtype=float)
    if "env_feedback" in df.columns:
        reward = df["env_feedback"].apply(lambda x: x.get("reward") if isinstance(x, dict) else np.nan)
        reward = pd.to_numeric(reward, errors="coerce").to_numpy(dtype=float)

    executable_action_result = _extract_series(df, "executable_action")
    if isinstance(executable_action_result, tuple):
        executable_action, executable_action_labels = executable_action_result
    else:
        executable_action = executable_action_result
        executable_action_labels = None
    if executable_action.ndim == 1:
        executable_action = np.stack([executable_action], axis=1)
    elif executable_action.ndim == 0:
        executable_action = np.full((n_frames, 1), float(executable_action), dtype=float)
    if executable_action_labels is None:
        executable_action_labels = ["action"] if executable_action.shape[1] == 1 else [f"a{i}" for i in range(executable_action.shape[1])]

    positions = None
    quats = None
    eulers = None
    omegas = None
    target = None
    if "env_feedback" in df.columns:
        raw_obs = df["env_feedback"].apply(lambda x: x.get("raw_obs", {}) if isinstance(x, dict) else {})
        positions = np.asarray(raw_obs.apply(lambda x: x.get("pos", [np.nan, np.nan, np.nan])).tolist(), dtype=float)
        quats = np.asarray(raw_obs.apply(lambda x: x.get("quat", [1.0, 0.0, 0.0, 0.0])).tolist(), dtype=float)
        eulers = np.asarray(raw_obs.apply(lambda x: x.get("euler", [np.nan, np.nan, np.nan])).tolist(), dtype=float)
        omegas = np.asarray(raw_obs.apply(lambda x: x.get("omega", [np.nan, np.nan, np.nan])).tolist(), dtype=float)
        targets = np.asarray(raw_obs.apply(lambda x: x.get("target_pos", [np.nan, np.nan, np.nan])).tolist(), dtype=float)
        target = targets[np.isfinite(targets).all(axis=1)][0] if np.isfinite(targets).all(axis=1).any() else None
    elif {"x", "y", "z"}.issubset(df.columns):
        positions = df[["x", "y", "z"]].to_numpy(dtype=float)
        if {"qw", "qx", "qy", "qz"}.issubset(df.columns):
            quats = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
        if {"target_x", "target_y", "target_z"}.issubset(df.columns):
            target = df[["target_x", "target_y", "target_z"]].iloc[0].to_numpy(dtype=float)

    if positions is None or not np.isfinite(positions).any():
        raise ValueError(f"No valid 3D positions found in log: {log_file}")
    valid = np.isfinite(positions).all(axis=1)
    positions = positions[valid]
    times = times[valid]
    ux, uy, output_x, output_y, reward = ux[valid], uy[valid], output_x[valid], output_y[valid], reward[valid]
    executable_action = executable_action[valid]
    if quats is not None and len(quats) == len(valid):
        quats = quats[valid]
    if eulers is not None and len(eulers) == len(valid):
        eulers = eulers[valid]
    if omegas is not None and len(omegas) == len(valid):
        omegas = omegas[valid]
    n_frames = len(positions)
    if n_frames == 0:
        raise ValueError(f"No valid 3D positions found in log: {log_file}")

    if eulers is None or not np.isfinite(eulers).any():
        eulers = np.full((n_frames, 3), np.nan, dtype=float)
        if quats is not None and quats.shape[1] >= 4:
            for i, quat in enumerate(quats):
                if np.isfinite(quat).all():
                    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                    pitch_x, roll_y, yaw_z = rot.as_euler("xyz", degrees=True)
                    eulers[i] = [pitch_x, roll_y, yaw_z]
    else:
        eulers = np.rad2deg(eulers) if np.nanmax(np.abs(eulers)) <= 2 * np.pi + 1e-3 else eulers
    if omegas is None or not np.isfinite(omegas).any():
        omegas = np.full((n_frames, 3), np.nan, dtype=float)

    target = np.asarray(target, dtype=float) if target is not None and np.isfinite(target).all() else positions[-1]
    current_frame = 0
    playback_pos = 0.0
    playing = False
    play_speed = 1.0

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes([0.06, 0.20, 0.68, 0.72], projection="3d")
    ax.set_title("Dart Flight Replay")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    bounds = np.vstack([positions, target[None, :], [positions[:, 0].mean(), positions[:, 1].mean(), 0.0]])
    mins = np.nanmin(bounds, axis=0)
    maxs = np.nanmax(bounds, axis=0)
    span = max(float(np.max(maxs - mins)), 1.0)
    mid = (maxs + mins) / 2.0
    pad = 0.15 * span
    ax.set_xlim(mid[0] - span / 2 - pad, mid[0] + span / 2 + pad)
    ax.set_ylim(mid[1] - span / 2 - pad, mid[1] + span / 2 + pad)
    ax.set_zlim(max(0.0, mid[2] - span / 2 - pad), mid[2] + span / 2 + pad)

    target_marker = ax.scatter(*target, color="green", s=180, marker="*", label="Target")
    ax.text(target[0], target[1], target[2] + 0.2, f"Target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})", color="green")
    x_grid, y_grid = np.meshgrid([mid[0] - span / 2 - pad, mid[0] + span / 2 + pad], [mid[1] - span / 2 - pad, mid[1] + span / 2 + pad])
    ax.plot_surface(x_grid, y_grid, np.zeros_like(x_grid), color="lightgray", alpha=0.25)

    traj_line, = ax.plot([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], color="orange", linewidth=2.5, label="Trajectory")
    body_line, = ax.plot([], [], [], color="crimson", linewidth=3.5, label="Dart Body")
    wing_line, = ax.plot([], [], [], color="navy", linewidth=2.2, label="Wing")
    target_line, = ax.plot([], [], [], color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Target Vector")
    arrow_scale = 0.8
    arrow_x = ax.quiver(*positions[0], arrow_scale, 0, 0, color="r")
    arrow_y = ax.quiver(*positions[0], 0, arrow_scale, 0, color="g")
    arrow_z = ax.quiver(*positions[0], 0, 0, arrow_scale, color="b")

    status_text = ax.text2D(0.02, 0.96, "Status: Paused | P: Play/Pause | R: Reset", transform=ax.transAxes, fontsize=10, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "gray"})
    info_ax = fig.add_axes([0.76, 0.70, 0.22, 0.22])
    info_ax.axis("off")
    info_text = info_ax.text(0.0, 1.0, "", va="top", fontsize=10, family="monospace", bbox={"facecolor": "#f7f7f7", "alpha": 0.95, "edgecolor": "#cccccc"})

    obs_ax = fig.add_axes([0.76, 0.62, 0.22, 0.12])
    obs_ax.set_title("Observation Features")
    obs_ax.set_xlabel("Time (s)")
    obs_ax.grid(True, alpha=0.25)
    obs_lines = [obs_ax.plot([], [], linewidth=1.4, label=label)[0] for label in ["ux", "uy", "output_x", "output_y"]]
    obs_ax.legend(loc="upper right", fontsize=7, ncol=2)

    action_ax = fig.add_axes([0.76, 0.46, 0.22, 0.12])
    action_ax.set_title("Executable Action")
    action_ax.set_xlabel("Time (s)")
    action_ax.grid(True, alpha=0.25)
    action_lines = [action_ax.plot([], [], linewidth=1.4, label=label)[0] for label in executable_action_labels]
    action_ax.legend(loc="upper right", fontsize=7, ncol=2)

    pose_ax = fig.add_axes([0.76, 0.30, 0.22, 0.12])
    pose_ax.set_title("Orientation Angles")
    pose_ax.set_xlabel("Time (s)")
    pose_ax.set_ylabel("Deg")
    pose_ax.grid(True, alpha=0.25)
    pose_lines = [pose_ax.plot([], [], linewidth=1.4, label=label)[0] for label in ["pitch_x", "roll_y", "yaw_z"]]
    pose_ax.legend(loc="upper right", fontsize=7)

    omega_ax = fig.add_axes([0.76, 0.14, 0.22, 0.12])
    omega_ax.set_title("Angular Velocity")
    omega_ax.set_xlabel("Time (s)")
    omega_ax.set_ylabel("rad/s")
    omega_ax.grid(True, alpha=0.25)
    omega_lines = [omega_ax.plot([], [], linewidth=1.4, label=label)[0] for label in ["omega_x", "omega_y", "omega_z"]]
    omega_ax.legend(loc="upper right", fontsize=7)

    ax_slider = plt.axes([0.12, 0.08, 0.56, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)
    speed_ax = plt.axes([0.12, 0.03, 0.56, 0.03])
    speed_slider = Slider(speed_ax, "Speed", 0.25, 5.0, valinit=1.0, valstep=0.25)

    def _set_axis_window(axis, x_vals, y_vals, min_abs=1.0):
        axis.set_xlim(0.0, max(float(times[-1]), 1.0))
        y_vals = np.asarray(y_vals, dtype=float)
        max_abs = float(np.nanmax(np.abs(y_vals))) if np.isfinite(y_vals).any() else min_abs
        axis.set_ylim(-max(min_abs, max_abs * 1.2), max(min_abs, max_abs * 1.2))

    def update_scene(frame):
        nonlocal arrow_x, arrow_y, arrow_z
        pos = positions[frame]
        traj_line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
        traj_line.set_3d_properties(positions[: frame + 1, 2])

        rot_mat = np.eye(3)
        if quats is not None and quats.shape[1] >= 4 and np.isfinite(quats[frame]).all():
            quat = quats[frame]
            rot_mat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        # 当前坐标系约定：x 右、y 前、z 上
        right = rot_mat @ np.array([1.0, 0.0, 0.0])
        forward = rot_mat @ np.array([0.0, 1.0, 0.0])
        up = rot_mat @ np.array([0.0, 0.0, 1.0])
        nose = pos + forward * 0.6
        tail = pos - forward * 0.4
        wing_left = pos - right * 0.35
        wing_right = pos + right * 0.35
        body_line.set_data([tail[0], nose[0]], [tail[1], nose[1]])
        body_line.set_3d_properties([tail[2], nose[2]])
        wing_line.set_data([wing_left[0], wing_right[0]], [wing_left[1], wing_right[1]])
        wing_line.set_3d_properties([wing_left[2], wing_right[2]])
        target_line.set_data([pos[0], target[0]], [pos[1], target[1]])
        target_line.set_3d_properties([pos[2], target[2]])

        for arr in [arrow_x, arrow_y, arrow_z]:
            arr.remove()
        arrow_x = ax.quiver(*pos, *(forward * arrow_scale), color="r")
        arrow_y = ax.quiver(*pos, *(right * arrow_scale), color="g")
        arrow_z = ax.quiver(*pos, *(up * arrow_scale), color="b")

        dist = float(np.linalg.norm(target - pos))
        status_text.set_text(f"Status: {'Playing' if playing else 'Paused'} | P: Play/Pause | R: Reset")
        info_text.set_text(
            f"Frame:    {frame + 1}/{n_frames}\n"
            f"Time:     {times[frame]:.2f} s\n"
            f"Speed:    {play_speed:.2f}x\n"
            f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            f"Target:   [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]\n"
            f"Distance: {dist:.3f} m\n"
            f"Reward:   {reward[frame]:.3f}"
        )

        t_plot = times[: frame + 1]
        obs_data = [ux[: frame + 1], uy[: frame + 1], output_x[: frame + 1], output_y[: frame + 1]]
        for line, y_data in zip(obs_lines, obs_data):
            line.set_data(t_plot, y_data)
        _set_axis_window(obs_ax, t_plot, np.concatenate(obs_data), min_abs=1.0)

        for i, line in enumerate(action_lines):
            line.set_data(t_plot, executable_action[: frame + 1, i])
        _set_axis_window(action_ax, t_plot, executable_action[: frame + 1].ravel(), min_abs=5.0)

        pose_data = [eulers[: frame + 1, 0], eulers[: frame + 1, 1], eulers[: frame + 1, 2]]
        for line, y_data in zip(pose_lines, pose_data):
            line.set_data(t_plot, y_data)
        _set_axis_window(pose_ax, t_plot, np.concatenate(pose_data), min_abs=5.0)

        omega_data = [omegas[: frame + 1, 0], omegas[: frame + 1, 1], omegas[: frame + 1, 2]]
        for line, y_data in zip(omega_lines, omega_data):
            line.set_data(t_plot, y_data)
        _set_axis_window(omega_ax, t_plot, np.concatenate(omega_data), min_abs=1.0)
        fig.canvas.draw_idle()

    def on_slider_change(val):
        nonlocal current_frame, playback_pos
        current_frame = int(val)
        playback_pos = float(current_frame)
        update_scene(current_frame)

    def on_speed_change(val):
        nonlocal play_speed
        play_speed = float(val)
        update_scene(current_frame)

    def on_key(event):
        nonlocal playing, current_frame, playback_pos
        if event.key and event.key.lower() == "p":
            playing = not playing
            playback_pos = float(current_frame)
            update_scene(current_frame)
        elif event.key and event.key.lower() == "r":
            playing = False
            current_frame = 0
            playback_pos = 0.0
            slider.set_val(0)

    def animate(_frame):
        nonlocal current_frame, playback_pos, playing
        if playing:
            playback_pos += play_speed
            if playback_pos >= n_frames - 1:
                playback_pos = float(n_frames - 1)
                playing = False
            current_frame = int(playback_pos)
            slider.set_val(current_frame)
        return tuple([traj_line, body_line, wing_line, target_line, target_marker, status_text, info_text, *obs_lines, *action_lines, *pose_lines, *omega_lines])

    slider.on_changed(on_slider_change)
    speed_slider.on_changed(on_speed_change)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update_scene(0)
    ani = animation.FuncAnimation(fig, animate, interval=30, blit=False)
    fig._ani = ani
    ax.legend(loc="upper right")
    plt.show()


def main() -> None:
    params = load_eval_params(__file__)
    env, runner, logger, env_cfg = _build_components(params)

    target_pos = np.asarray(env_cfg.target_pos, dtype=np.float32)
    raw_obs = env.reset()
    raw_obs.target_pos = target_pos.copy()
    done = False
    truncated = False

    while not done and not truncated:
        raw_obs, reward, done, truncated = runner.run_one_step(raw_obs)

    logger.flush_summary()
    env.close()
    print(f"Single rollout finished. Log saved to {logger._step_path}")
    _plot_log(str(logger._step_path))


if __name__ == "__main__":
    main()
