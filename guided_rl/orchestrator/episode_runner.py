from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guided_rl.common.enums import DataRequest, DataTag
from guided_rl.common.interfaces import ActionIntegrator, BaseActionMaker, Environment, ObservationProcessor, RLActionMaker, StepLogger
from guided_rl.common.types import EnvFeedback
from guided_rl.runtime.data_hub import DataHub


@dataclass
class EpisodeRunner:
    env: Environment
    data_hub: DataHub
    observation_processor: ObservationProcessor
    base_action_maker: BaseActionMaker
    rl_action_maker: RLActionMaker
    action_integrator: ActionIntegrator
    logger: StepLogger | None = None

    def run_one_step(self, raw_obs):
        self.data_hub.begin_step()
        self.data_hub.input(DataTag.RAW_OBS, raw_obs)

        obs_input = self.data_hub.output(DataRequest.OBSERVATION_INPUT)
        obs_features = self.observation_processor.compute(obs_input)
        self.data_hub.input(DataTag.OBS_FEATURES, obs_features)

        decision_pack1 = self.data_hub.output(DataRequest.DECISION_PACK1)
        base_action = self.base_action_maker.compute(decision_pack1)
        self.data_hub.input(DataTag.BASE_ACTION, base_action)

        decision_pack2 = self.data_hub.output(DataRequest.DECISION_PACK2)
        rl_action = self.rl_action_maker.compute(decision_pack2)
        self.data_hub.input(DataTag.RL_ACTION, rl_action)

        action_input = self.data_hub.output(DataRequest.ACTION_INTEGRATION_INPUT)
        executable_action = self.action_integrator.integrate(action_input)
        self.data_hub.input(DataTag.EXECUTABLE_ACTION, executable_action)

        env_action = np.asarray(
            [executable_action.ul, executable_action.ur, executable_action.dl, executable_action.dr],
            dtype=np.float32,
        )
        next_raw_obs, reward, terminated, truncated = self.env.step(env_action)
        feedback = EnvFeedback(
            raw_obs=next_raw_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        self.data_hub.input(DataTag.ENV_FEEDBACK, feedback)

        step_record = self.data_hub.output(DataRequest.STEP_RECORD)
        if self.logger is not None:
            self.logger.log_step(step_record)

        self.data_hub.commit_step()
        return next_raw_obs, reward, terminated, truncated
