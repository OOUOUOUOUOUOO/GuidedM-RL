from __future__ import annotations

from enum import Enum, auto


class DataTag(Enum):
    RAW_OBS = auto()
    OBS_FEATURES = auto()
    DECISION_PACK1 = auto()
    BASE_ACTION = auto()
    DECISION_PACK2 = auto()
    RL_ACTION = auto()
    ACTION_INTEGRATION_INPUT = auto()
    EXECUTABLE_ACTION = auto()
    ENV_FEEDBACK = auto()


class DataRequest(Enum):
    RAW_OBS = auto()
    OBSERVATION_INPUT = auto()
    OBS_FEATURES = auto()
    DECISION_PACK1 = auto()
    BASE_ACTION = auto()
    DECISION_PACK2 = auto()
    RL_ACTION = auto()
    ACTION_INTEGRATION_INPUT = auto()
    EXECUTABLE_ACTION = auto()
    STEP_RECORD = auto()
    HISTORY = auto()
