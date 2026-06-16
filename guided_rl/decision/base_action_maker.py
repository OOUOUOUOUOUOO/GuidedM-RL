from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guided_rl.common.types import BaseAction, BaseActionMakerConfig, DecisionPack1


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


@dataclass
class PDController:
    kp: float
    kd: float
    out_limit: float = 666.0

    last_error: float = 0.0

    def reset(self) -> None:
        self.last_error = 0.0

    def update(self, target: float, measure: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        error = target - measure
        p_out = self.kp * error
        d_out = self.kd * (error - self.last_error) / dt

        self.last_error = error

        out = p_out + d_out
        return clamp(out, -self.out_limit, self.out_limit)


class BaseActionMaker:
    def __init__(self, config: BaseActionMakerConfig):
        self.config = config
        self.roll_angle_pid = PDController(12.0, 3.0, out_limit=666.0)
        self.roll_rate_pid = PDController(5.0, 0.0, out_limit=30.0)
        self.pitch_angle_pid = PDController(100.0, 0.0, out_limit=666.0)
        self.vis_x_pid = PDController(10.0, 5.0, out_limit=666.0)
        self.pitch_rate_pid = PDController(10.0, 0.37, out_limit=30.0)
        self.yaw_rate_pid = PDController(10.0, 1.0, out_limit=30.0)
        self.last_output = BaseAction(0.0, 0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.roll_angle_pid.reset()
        self.roll_rate_pid.reset()
        self.pitch_angle_pid.reset()
        self.vis_x_pid.reset()
        self.pitch_rate_pid.reset()
        self.yaw_rate_pid.reset()
        self.last_output = BaseAction(0.0, 0.0, 0.0, 0.0)

    def compute(self, decision_pack1: DecisionPack1, dt: float = 0.01) -> BaseAction:
        if not decision_pack1.visible:
            return self.last_output

        vis_x = float(decision_pack1.output_x)
        vis_y = float(decision_pack1.output_y)
        pitch = float(decision_pack1.pitch)
        roll = float(decision_pack1.roll)
        yaw = float(decision_pack1.yaw)
        gyro_pitch = float(decision_pack1.gyro_pitch)
        gyro_roll = float(decision_pack1.gyro_roll)
        gyro_yaw = float(decision_pack1.gyro_yaw)
        time = float(decision_pack1.time)

        roll_target_rate = self.roll_angle_pid.update(0, roll, dt)
        roll_output = self.roll_rate_pid.update(roll_target_rate, gyro_roll, dt)

        target_yaw_rate = self.vis_x_pid.update(0.0, vis_x, dt)
        yaw_output = self.yaw_rate_pid.update(target_yaw_rate, gyro_yaw, dt)

        pitch_target_rate = self.pitch_angle_pid.update(0.0, -vis_y, dt)
        pitch_output = self.pitch_rate_pid.update(pitch_target_rate, gyro_pitch, dt)

        # ul = clamp(pitch_output - yaw_output + roll_output, -30.0, 30.0)
        # ur = clamp(-pitch_output - yaw_output + roll_output, -30.0, 30.0)
        # dl = clamp(pitch_output + yaw_output + roll_output, -30.0, 30.0)
        # dr = clamp(-pitch_output + yaw_output + roll_output, -30.0, 30.0)

        
        if time < 3.0:
            ul = clamp(-yaw_output + roll_output  , -30.0, 30.0)
            ur = clamp(-yaw_output + roll_output , -30.0, 30.0)
            dl = clamp(yaw_output + roll_output  , -30.0, 30.0)
            dr = clamp(yaw_output + roll_output  , -30.0, 30.0)
            # ul = 0.0
            # ur = 0.0
            # dl = 0.0
            # dr = 0.0
        else:
            ul = clamp(pitch_output , -30.0, 30.0)
            ur = clamp(-pitch_output  , -30.0, 30.0)
            dl = clamp(pitch_output  , -30.0, 30.0)
            dr = clamp(-pitch_output , -30.0, 30.0)


        action = BaseAction(ul=ul, ur=ur, dl=dl, dr=dr)
        self.last_output = action
        return action


__all__ = ["BaseActionMaker", "PDController"]
