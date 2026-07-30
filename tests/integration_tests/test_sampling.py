import pytest

from minipcn import Sampler


def test_sampling(rng, log_target_fn, step_fn, dims, xp):
    x_init = rng.normal(size=(100, dims))  # Initial samples

    sampler = Sampler(
        log_prob_fn=log_target_fn,
        dims=dims,
        step_fn=step_fn,
        target_acceptance_rate=0.234,
        xp=xp,
    )

    chain, history = sampler.sample(x_init, n_steps=100, rng=rng)
    assert chain.shape == (101, 100, dims)
    assert history.it[-1] == 99


@pytest.mark.parametrize("step_fn", ["pCN", "tpCN"])
def test_sampling_jax_jit_return_last_only(step_fn):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from orng.functional import create_functional_backend

    dims = 4
    rng_backend = create_functional_backend("jax")
    rng_state = rng_backend.init_state(seed=42, generator=None)
    x_init, rng_state = rng_backend.normal(
        rng_state,
        loc=0.0,
        scale=1.0,
        size=(32, dims),
        dtype=jnp.float32,
    )

    def log_target_fn(x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    sampler = Sampler(
        log_prob_fn=log_target_fn,
        dims=dims,
        step_fn=step_fn,
        target_acceptance_rate=0.234,
        xp=jnp,
    )

    @jax.jit
    def run(x0, state):
        samples, history, _ = sampler.sample_functional(
            x0,
            n_steps=8,
            rng_state=state,
            verbose=False,
            return_last_only=True,
        )
        return samples, history

    final_samples, history = run(x_init, rng_state)
    assert final_samples.shape == x_init.shape
    assert final_samples.dtype == x_init.dtype
    assert history.it.shape == (8,)
    assert history.acceptance_rate.shape == (8,)


def test_sampling_with_functional_backend_input():
    pytest.importorskip("numpy")
    import numpy as np
    from orng.functional import create_functional_backend

    dims = 2
    backend = create_functional_backend("numpy")
    rng_state = backend.init_state(seed=42, generator=None)
    x_init, rng_state = backend.normal(
        rng_state,
        loc=0.0,
        scale=1.0,
        size=(16, dims),
        dtype=np.float64,
    )

    def log_target_fn(x):
        return -0.5 * np.sum(x**2, axis=-1)

    sampler = Sampler(
        log_prob_fn=log_target_fn,
        dims=dims,
        step_fn="pCN",
        target_acceptance_rate=0.234,
        xp=np,
    )

    final_samples, history, next_rng_state = sampler.sample_functional(
        x_init,
        n_steps=4,
        rng_state=rng_state,
        verbose=False,
        return_last_only=True,
    )
    assert final_samples.shape == x_init.shape
    assert history.it[-1] == 3
    assert next_rng_state is not None


def test_sampling_with_adaptation_disabled(rng, log_target_fn, dims, xp):
    x_init = rng.normal(size=(100, dims))

    sampler = Sampler(
        log_prob_fn=log_target_fn,
        dims=dims,
        step_fn="pCN",
        target_acceptance_rate=0.234,
        xp=xp,
        adaptive=False,
    )

    chain, history = sampler.sample(x_init, n_steps=8, rng=rng, verbose=False)

    assert chain.shape == (9, 100, dims)
    assert history.it[-1] == 7
    assert "rho" not in history.extra_stats


def test_sampling_jax_jit_with_functional_backend_input():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from orng.functional import create_functional_backend

    dims = 4
    backend = create_functional_backend("jax")
    rng_state = backend.init_state(seed=42, generator=None)
    x_init, rng_state = backend.normal(
        rng_state,
        loc=0.0,
        scale=1.0,
        size=(32, dims),
        dtype=jnp.float32,
    )

    def log_target_fn(x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    sampler = Sampler(
        log_prob_fn=log_target_fn,
        dims=dims,
        step_fn="pCN",
        target_acceptance_rate=0.234,
        xp=jnp,
    )

    @jax.jit
    def run(x0, state):
        samples, history, _ = sampler.sample_functional(
            x0,
            n_steps=8,
            rng_state=state,
            verbose=False,
            return_last_only=True,
        )
        return samples, history

    final_samples, history = run(x_init, rng_state)
    assert final_samples.shape == x_init.shape
    assert final_samples.dtype == x_init.dtype
    assert history.it.shape == (8,)
    assert history.acceptance_rate.shape == (8,)
