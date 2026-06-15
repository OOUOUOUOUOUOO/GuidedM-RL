from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import ActionIntegrationInput, BaseAction, DecisionPack1, DecisionPack2, ExecutableAction, ObservationFeatures, RawEnvObs, RLAction, StepData


class ObservationProcessor(Protocol):
    def compute(self, obs: RawEnvObs) -> ObservationFeatures:
        ...


class BaseActionMaker(Protocol):
    def compute(self, pack: DecisionPack1) -> BaseAction:
        ...


class RLActionMaker(Protocol):
    def compute(self, pack: DecisionPack2) -> RLAction:
        ...


class ActionIntegrator(Protocol):
    def integrate(self, action_input: ActionIntegrationInput) -> ExecutableAction:
        ...


class Environment(Protocol):
    def reset(self) -> tuple[RawEnvObs, dict]:
        ...

    def step(self, action: np.ndarray) -> tuple[RawEnvObs, float, bool, bool, dict]:
        ...


class StepLogger(Protocol):
    def log_step(self, step_record: StepData) -> None:
        ...

    def flush_summary(self) -> None:
        ...
