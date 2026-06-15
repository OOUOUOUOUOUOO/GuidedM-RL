from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from guided_rl.common.enums import DataRequest, DataTag
from guided_rl.common.types import (
    ActionIntegrationInput,
    BaseAction,
    DecisionPack1,
    DecisionPack2,
    EnvFeedback,
    ExecutableAction,
    ObservationFeatures,
    RawEnvObs,
    RLAction,
    StepData,
)


class DataHub:
    def __init__(self, history_len: int = 3):
        self.history_len = int(history_len)
        self.history: deque[StepData] = deque(maxlen=self.history_len)
        self.current = StepData()

    def reset_episode(self) -> None:
        self.history.clear()
        self.current = StepData()

    def begin_step(self) -> None:
        self.current = StepData()

    def input(self, tag: DataTag, data: Any) -> None:
        if tag == DataTag.RAW_OBS:
            self.current.raw_obs = data
        elif tag == DataTag.OBS_FEATURES:
            self.current.obs_features = data
        elif tag == DataTag.BASE_ACTION:
            self.current.base_action = data
        elif tag == DataTag.RL_ACTION:
            self.current.rl_action = data
        elif tag == DataTag.EXECUTABLE_ACTION:
            self.current.executable_action = data
        elif tag == DataTag.ENV_FEEDBACK:
            self.current.env_feedback = data
        else:
            raise ValueError(f"Unsupported DataTag: {tag}")

    def output(self, request: DataRequest):
        if request == DataRequest.OBSERVATION_INPUT:
            return self._build_observation_input()
        if request == DataRequest.DECISION_PACK1:
            pack = self._build_decision_pack1()
            self.current.decision_pack1 = pack
            return pack
        if request == DataRequest.DECISION_PACK2:
            pack = self._build_decision_pack2()
            self.current.decision_pack2 = pack
            return pack
        if request == DataRequest.ACTION_INTEGRATION_INPUT:
            action_input = self._build_action_integration_input()
            self.current.action_input = action_input
            return action_input
        if request == DataRequest.EXECUTABLE_ACTION:
            return self.current.executable_action
        if request == DataRequest.STEP_RECORD:
            return self._build_step_record()
        if request == DataRequest.HISTORY:
            return list(self.history)
        raise ValueError(f"Unsupported DataRequest: {request}")

    def commit_step(self) -> None:
        self.history.append(self.current)

    def _build_observation_input(self) -> RawEnvObs:
        return self.current.raw_obs

    def _build_decision_pack1(self) -> DecisionPack1:
        return DecisionPack1(
            output_x=self.current.obs_features.output_x,
            output_y=self.current.obs_features.output_y,
            visible=self.current.obs_features.visible,
            pitch=self.current.raw_obs.euler[0],
            roll=self.current.raw_obs.euler[1],
            yaw=self.current.raw_obs.euler[2],
            gyro_pitch=self.current.raw_obs.omega[0],
            gyro_roll=self.current.raw_obs.omega[1],
            gyro_yaw=self.current.raw_obs.omega[2],
            time=float(self.current.raw_obs.time),
        )

    def _build_decision_pack2(self) -> DecisionPack2:
        if self.current.base_action is None or self.current.raw_obs is None or self.current.obs_features is None:
            raise ValueError("raw_obs, obs_features and base_action must be available before building decision pack2")
        return DecisionPack2(
            output_x=float(self.current.obs_features.output_x),
            output_y=float(self.current.obs_features.output_y),
            base_action_ul=float(self.current.base_action.ul),
            base_action_ur=float(self.current.base_action.ur),
            base_action_dl=float(self.current.base_action.dl),
            base_action_dr=float(self.current.base_action.dr),
            quat=np.asarray(self.current.raw_obs.quat, dtype=np.float32).copy(),
            omega=np.asarray(self.current.raw_obs.omega, dtype=np.float32).copy(),
            time=float(self.current.raw_obs.time),
        )

    def _build_action_integration_input(self) -> ActionIntegrationInput:
        if self.current.base_action is None or self.current.rl_action is None:
            raise ValueError("base_action and rl_action must be available before integration")
        return ActionIntegrationInput(base_action=self.current.base_action, rl_action=self.current.rl_action)
    
    def _build_step_record(self) -> StepData:
        return StepData(
            raw_obs=self.current.raw_obs,
            obs_features=self.current.obs_features,
            base_action=self.current.base_action,
            rl_action=self.current.rl_action,
            executable_action=self.current.executable_action,
            env_feedback=self.current.env_feedback,
            decision_pack1=self.current.decision_pack1,
            decision_pack2=self.current.decision_pack2,
            action_input=self.current.action_input,
        )


__all__ = ["DataHub"]
