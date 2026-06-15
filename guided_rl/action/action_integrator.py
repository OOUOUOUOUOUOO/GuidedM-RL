from __future__ import annotations

import numpy as np

from guided_rl.common.types import ActionIntegrationInput, ActionIntegratorConfig, ExecutableAction


class ActionIntegrator:
    def __init__(self, config: ActionIntegratorConfig):
        self.config = config

    def integrate(self, action_input: ActionIntegrationInput) -> ExecutableAction:
        base = np.asarray(
            [action_input.base_action.ul, action_input.base_action.ur, action_input.base_action.dl, action_input.base_action.dr],
            dtype=np.float32,
        ).reshape(-1)
        rl = np.asarray(
            [action_input.rl_action.ul, action_input.rl_action.ur, action_input.rl_action.dl, action_input.rl_action.dr],
            dtype=np.float32,
        ).reshape(-1)
        if base.shape != (4,):
            raise ValueError(f"base_action must have shape (4,), got {base.shape}")
        if rl.shape != (4,):
            raise ValueError(f"rl_action must have shape (4,), got {rl.shape}")

        final = base + rl
        final = np.clip(final, -self.config.servo_limit_abs, self.config.servo_limit_abs)
        ul = final[0]
        ur = final[1]
        dl = final[2]
        dr = final[3]
        return ExecutableAction(ul=ul, ur=ur, dl=dl, dr=dr)
