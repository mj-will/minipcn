from unittest.mock import MagicMock

import numpy as np
import pytest
from orng import RandomGenerator

from minipcn.sampler import Sampler
from minipcn.step import PCNStep, StepState
from minipcn.utils import ChainState, to_numpy_array


@pytest.fixture
def sampler():
    return MagicMock(spec=Sampler, rng=None)


def test_sampler_rejects_step_instances(log_target_fn):
    step = PCNStep(dims=2, xp=np, rng_backend=object())

    with pytest.raises(TypeError, match="step_fn must be the name"):
        Sampler(log_prob_fn=log_target_fn, dims=2, step_fn=step, xp=np)


def test_resolve_rng_none_uses_seeded_numpy_backend(sampler):
    seed = 1234
    rng_backend, rng_state = Sampler._resolve_rng(sampler, None, seed=seed)
    rng_backend_2, rng_state_2 = Sampler._resolve_rng(sampler, None, seed=seed)

    u, next_state = rng_backend.random(rng_state, size=(5,), dtype=None)
    u2, next_state_2 = rng_backend_2.random(rng_state_2, size=(5,), dtype=None)

    assert rng_backend is not None
    assert rng_state is not None
    assert next_state is not None
    assert next_state_2 is not None
    np.testing.assert_allclose(u, u2)


def test_resolve_rng_random_generator(backend, sampler):
    rng = RandomGenerator(backend=backend, seed=1234)
    rng_2 = RandomGenerator(backend=backend, seed=1234)

    rng_backend, rng_state = Sampler._resolve_rng(sampler, rng, seed=None)
    rng_backend_2, rng_state_2 = Sampler._resolve_rng(
        sampler, rng_2, seed=None
    )

    u, next_state = rng_backend.random(rng_state, size=(5,), dtype=None)
    u2, next_state_2 = rng_backend_2.random(rng_state_2, size=(5,), dtype=None)

    assert rng_backend is not None
    assert rng_state is not None
    assert next_state is not None
    assert next_state_2 is not None
    np.testing.assert_allclose(to_numpy_array(u), to_numpy_array(u2))


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
