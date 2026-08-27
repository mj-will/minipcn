import numpy as np
import pytest

from minipcn import Sampler


def _make_jax_sampler(step_fn):
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
    return jax, jnp, sampler, x_init, rng_state


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
    jax, _, sampler, x_init, rng_state = _make_jax_sampler(step_fn)

    def run(x0, state):
        samples, history, _ = sampler.sample_functional(
            x0,
            n_steps=8,
            rng_state=state,
            verbose=False,
            return_last_only=True,
        )
        return samples, history

    jaxpr = jax.make_jaxpr(run)(x_init, rng_state)
    scan_equation = next(
        equation
        for equation in jaxpr.jaxpr.eqns
        if equation.primitive.name == "scan"
    )
    particle_history_shape = (8, *x_init.shape)
    assert particle_history_shape not in {
        variable.aval.shape for variable in scan_equation.outvars
    }

    final_samples, history = jax.jit(run)(x_init, rng_state)
    assert final_samples.shape == x_init.shape
    assert final_samples.dtype == x_init.dtype
    assert history.it.shape == (8,)
    assert history.acceptance_rate.shape == (8,)


def test_sampling_jax_jit_scan_can_be_disabled():
    jax, _, sampler, x_init, rng_state = _make_jax_sampler("pCN")

    def run(x0, state):
        return sampler.sample_functional(
            x0,
            n_steps=8,
            rng_state=state,
            verbose=False,
            return_last_only=True,
            use_scan=False,
        )

    jaxpr = jax.make_jaxpr(run)(x_init, rng_state)
    assert "scan" not in {
        equation.primitive.name for equation in jaxpr.jaxpr.eqns
    }

    final_samples, history, _ = jax.jit(run)(x_init, rng_state)
    assert final_samples.shape == x_init.shape
    assert history.it.shape == (8,)


@pytest.mark.parametrize("step_fn", ["pCN", "tpCN"])
def test_sampling_jax_scan_matches_loop(step_fn):
    jax, _, sampler, x_init, rng_state = _make_jax_sampler(step_fn)

    scan_chain, scan_history, scan_rng_state = sampler.sample_functional(
        x_init,
        n_steps=8,
        rng_state=rng_state,
        verbose=False,
        use_scan=True,
    )
    loop_chain, loop_history, loop_rng_state = sampler.sample_functional(
        x_init,
        n_steps=8,
        rng_state=rng_state,
        verbose=False,
        use_scan=False,
    )

    np.testing.assert_allclose(scan_chain, loop_chain, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(scan_history.it, loop_history.it)
    np.testing.assert_allclose(
        scan_history.acceptance_rate,
        loop_history.acceptance_rate,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        scan_history.target_acceptance_rate,
        loop_history.target_acceptance_rate,
        rtol=1e-5,
        atol=1e-6,
    )
    assert scan_history.step == loop_history.step
    assert scan_history.extra_stats.keys() == loop_history.extra_stats.keys()
    for key in scan_history.extra_stats:
        np.testing.assert_allclose(
            scan_history.extra_stats[key],
            loop_history.extra_stats[key],
            rtol=1e-5,
            atol=1e-6,
        )
    np.testing.assert_array_equal(
        jax.random.key_data(scan_rng_state),
        jax.random.key_data(loop_rng_state),
    )


def test_forced_scan_rejects_verbose_output():
    _, _, sampler, x_init, rng_state = _make_jax_sampler("pCN")

    with pytest.raises(
        ValueError, match="use_scan=True requires verbose=False"
    ):
        sampler.sample_functional(
            x_init,
            n_steps=8,
            rng_state=rng_state,
            use_scan=True,
        )


def test_sampling_with_functional_backend_input():
    pytest.importorskip("numpy")
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
