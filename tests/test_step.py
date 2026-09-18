"""Direct step behavior."""

import numpy as np

from minipcn.step import PCNStep, StepState
from minipcn.utils import ChainState


def test_pcn_step_adapt_can_be_disabled():
    step = PCNStep(dims=2, xp=np, rng_backend=object(), adaptive=False)
    state = StepState(
        mu=np.zeros(2),
        cov=np.eye(2),
        inv_cov=np.eye(2),
        chol_cov=np.eye(2),
        rho=np.asarray(0.5),
    )
    chain_state = ChainState(
        it=0,
        acceptance_rate=np.asarray(0.9),
        target_acceptance_rate=0.234,
        step=step.step_name,
    )

    next_state, next_chain_state = step.adapt(
        state,
        chain_state,
        samples=np.zeros((4, 2)),
    )

    assert next_state is state
    assert next_chain_state is chain_state
