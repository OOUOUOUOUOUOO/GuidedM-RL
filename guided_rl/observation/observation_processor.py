from __future__ import annotations

import math

import numpy as np

from guided_rl.common.types import ObservationFeatures, ObservationProcessorConfig, RawEnvObs


class ObservationProcessor:
    def __init__(self, config: ObservationProcessorConfig):
        self.config = config
        self._parabola_cache: dict[str, float] | None = None

    def _build_yz_parabola(self, obs: RawEnvObs) -> dict[str, float]:
        pos = np.asarray(obs.pos, dtype=np.float32).reshape(-1)
        vel = np.asarray(obs.vel, dtype=np.float32).reshape(-1)
        target_pos = np.asarray(obs.target_pos, dtype=np.float32).reshape(-1)

        if pos.shape[0] != 3 or vel.shape[0] != 3 or target_pos.shape[0] != 3:
            raise ValueError("pos, vel and target_pos must have shape (3,)")

        y0 = float(pos[1])
        z0 = float(pos[2])
        vy0 = float(vel[1])
        vz0 = float(vel[2])
        y_target = float(target_pos[1])
        z_target = float(target_pos[2])

        # 这里用 YZ 平面拟合一条抛物线 z(y) = a (y-y0)^2 + b (y-y0) + z0
        # 并强制它经过 (y_target, z_target)
        if abs(y_target - y0) <= 1e-6:
            return {
                "y0": y0,
                "z0": z0,
                "a": 0.0,
                "b": 0.0 if abs(vy0) <= 1e-6 else vz0 / vy0,
            }

        b = 0.0 if abs(vy0) <= 1e-6 else vz0 / vy0
        dy_target = y_target - y0
        a = (z_target - z0 - b * dy_target) / (dy_target * dy_target)

        return {"y0": y0, "z0": z0, "a": float(a), "b": float(b)}

    def _parabola_height(self, y: float) -> float:
        if self._parabola_cache is None:
            return 0.0
        dy = y - self._parabola_cache["y0"]
        return float(
            self._parabola_cache["z0"]
            + self._parabola_cache["b"] * dy
            + self._parabola_cache["a"] * dy * dy
        )

    def compute(self, obs: RawEnvObs) -> ObservationFeatures:
        # 世界系约定：
        # X: right  右
        # Y: forward 前
        # Z: up     上
        rel_pos = (
            np.asarray(obs.target_pos, dtype=np.float32)
            - np.asarray(obs.pos, dtype=np.float32)
        )

        if rel_pos.shape[0] != 3:
            raise ValueError("target_pos and pos must have shape (3,)")

        vel = np.asarray(obs.vel, dtype=np.float32).reshape(-1)
        if vel.shape[0] != 3:
            raise ValueError("RawEnvObs.vel must have shape (3,)")

        if int(obs.step) == 1 or self._parabola_cache is None:
            self._parabola_cache = self._build_yz_parabola(obs)

        lateral = float(rel_pos[0])
        depth = float(rel_pos[1])
        vertical = float(rel_pos[2])

        fx = max(float(self.config.fx), 1e-6)
        fy = max(float(self.config.fy), 1e-6)
        depth_safe = max(depth, self.config.depth_proj_min)

        ux = float(
            np.clip(
                fx * lateral / depth_safe,
                -self.config.uv_err_clip,
                self.config.uv_err_clip,
            )
        )
        uy = float(
            np.clip(
                -fy * vertical / depth_safe,
                -self.config.uv_err_clip,
                self.config.uv_err_clip,
            )
        )

        if abs(depth) <= 1e-6 and abs(lateral) <= 1e-6:
            angle_x = 0.0
        else:
            angle_x = float(math.atan2(lateral, depth))

        parabola_z = self._parabola_height(obs.pos[1])
        output_y = float(obs.pos[2] - parabola_z)

        visible = bool(
            depth >= self.config.vis_depth_min
            and abs(math.degrees(math.atan2(lateral, max(depth, 1e-6))))
            <= self.config.vis_hfov_deg * 0.5
            and abs(math.degrees(math.atan2(vertical, max(depth, 1e-6))))
            <= self.config.vis_vfov_deg * 0.5
        )

        return ObservationFeatures(
            ux=ux,
            uy=uy,
            angle_x=angle_x,
            angle_y=output_y,
            output_x=float(depth_safe * math.tan(angle_x)),
            output_y=output_y,
            depth=depth,
            visible=visible,
            info={
                "target_step": int(obs.step),
                "target_time": float(obs.time),
                "rel_pos_world": rel_pos.tolist(),
                "vel_world": vel.tolist(),
                "parabola_cache": dict(self._parabola_cache) if self._parabola_cache is not None else None,
                "parabola_z": parabola_z,
            },
        )
