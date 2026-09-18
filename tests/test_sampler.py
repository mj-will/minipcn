from __future__ import annotations

import builtins
import pickle
import subprocess
import sys
from copy import deepcopy
from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from array_api_compat import array_namespace
from orng import RandomGenerator

from minipcn.sampler import Sampler
from minipcn.step import PCNStep, StepState, TPCNStep
from minipcn.utils import to_numpy_array


@pytest.fixture
def sampler():
    return MagicMock(spec=Sampler, rng=None, xp=np)


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


def test_resolve_rng_random_generator(backend, xp, sampler):
    sampler.xp = xp
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


def assert_states_equal(actual: StepState, expected: StepState) -> None:
    for field in fields(StepState):
        a = getattr(actual, field.name)
        b = getattr(expected, field.name)
        if b is None:
            assert a is None
        else:
            np.testing.assert_allclose(to_numpy_array(a), to_numpy_array(b))


@pytest.mark.parametrize("step_name", ["pcn", "tpcn"])
def test_init_fits_without_target_or_random_work(step_name: str) -> None:
    def target(x: Any) -> Any:
        pytest.fail("Fitting must not evaluate the target.")

    # Neither the target nor the deprecated sampler RNG participates in fitting.
    torch = pytest.importorskip("torch")
    sampler = Sampler(target, step_name, dims=2, xp=np)
    sampler.rng = object()
    x = torch.asarray(np.random.default_rng(2).normal(size=(64, 2)))
    state = sampler.init_step_state(x)
    assert state.mu.shape == (2,)
    assert state.chol_cov.shape == (2, 2)
    assert array_namespace(state.mu) is array_namespace(x)
    assert sampler.xp is np
    assert state.iteration == 0
    assert (state.nu is None) == (step_name == "pcn")


def test_split_eager_run_matches_uninterrupted(backend: str, xp: Any) -> None:
    x = xp.asarray(np.random.default_rng(12).normal(size=(64, 2)))
    sampler = Sampler(
        lambda x: -0.5 * xp.sum(x**2, axis=-1),
        "pcn",
        dims=2,
        xp=xp,
        adaptive=True,
    )
    initial = sampler.init_step_state(x)
    original = deepcopy(initial)
    rng = RandomGenerator(backend=backend, seed=42)
    whole, whole_history, whole_state = sampler.sample(
        x,
        5,
        rng=RandomGenerator(backend=backend, seed=42),
        verbose=False,
        step_state=initial,
        return_step_state=True,
    )
    first, first_history, first_state = sampler.sample(
        x,
        2,
        rng=rng,
        verbose=False,
        step_state=initial,
        return_step_state=True,
    )
    second, second_history, second_state = sampler.sample(
        first[-1],
        3,
        rng=rng,
        verbose=False,
        step_state=first_state,
        return_step_state=True,
    )
    np.testing.assert_array_equal(
        to_numpy_array(first), to_numpy_array(whole[:3])
    )
    np.testing.assert_array_equal(
        to_numpy_array(second), to_numpy_array(whole[2:])
    )
    assert first_history.it == [0, 1]
    assert second_history.it == [2, 3, 4]
    np.testing.assert_array_equal(
        first_history.acceptance_rate + second_history.acceptance_rate,
        whole_history.acceptance_rate,
    )
    assert_states_equal(second_state, whole_state)
    assert_states_equal(initial, original)
    assert first_state.iteration == 2
    assert second_state.iteration == 5


def test_supplied_state_skips_fit_and_evaluates_new_particles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated = []

    def target(x: Any) -> Any:
        evaluated.append(np.array(x, copy=True))
        return -0.5 * np.sum(x**2, axis=-1)

    sampler = Sampler(target, "tpcn", dims=2, adaptive=False)
    x = np.random.default_rng(7).normal(size=(64, 2))
    state = sampler.init_step_state(x)
    original = deepcopy(state)

    def fail(*args: Any) -> Any:
        pytest.fail("The supplied step state should bypass fitting.")

    monkeypatch.setattr(TPCNStep, "init_state", fail)
    # An external move changed the population. Reuse the fit, never a stale
    # log target from an earlier sampling call.
    changed = x + 3
    _, _, final = sampler.sample(
        changed,
        1,
        seed=9,
        step_state=state,
        verbose=False,
        return_step_state=True,
    )
    assert_states_equal(state, original)
    assert final.iteration == 1
    original.iteration = 1
    assert_states_equal(final, original)
    assert len(evaluated) == 2
    np.testing.assert_array_equal(evaluated[0], changed)


def test_jitted_initialization_and_functional_continuation() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.asarray(np.random.default_rng(14).normal(size=(64, 2)))
    sampler = Sampler(
        lambda x: -0.5 * jnp.sum(x**2, axis=-1),
        "tpcn",
        dims=2,
        xp=jnp,
        adaptive=True,
    )
    initial = jax.jit(sampler.init_step_state)(x)

    @jax.jit
    def segment(x: Any, key: Any, state: StepState) -> Any:
        return sampler.sample_functional(
            x,
            2,
            rng_state=key,
            step_state=state,
            return_step_state=True,
            return_last_only=True,
            verbose=False,
        )

    @jax.jit
    def full(x: Any, key: Any, state: StepState) -> Any:
        return sampler.sample_functional(
            x,
            4,
            rng_state=key,
            step_state=state,
            return_step_state=True,
            return_last_only=True,
            verbose=False,
        )

    key = jax.random.key(31)
    a, h_a, key_a, state_a = segment(x, key, initial)
    b, h_b, key_b, state_b = segment(a, key_a, state_a)
    expected, history, expected_key, expected_state = full(x, key, initial)
    np.testing.assert_allclose(b, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(
        jax.random.key_data(key_b), jax.random.key_data(expected_key)
    )
    np.testing.assert_allclose(state_b.rho, expected_state.rho)
    np.testing.assert_array_equal(
        jnp.concatenate((h_a.it, h_b.it)), history.it
    )
    assert int(state_b.iteration) == 4
    assert int(initial.iteration) == 0


def test_zero_steps_preserves_state() -> None:
    sampler = Sampler(lambda x: -0.5 * np.sum(x**2, axis=-1), "pcn", dims=2)
    x = np.random.default_rng(1).normal(size=(64, 2))
    initial = sampler.init_step_state(x)
    last, history, state = sampler.sample(
        x,
        0,
        step_state=initial,
        seed=4,
        verbose=False,
        return_last_only=True,
        return_step_state=True,
    )
    np.testing.assert_array_equal(last, x)
    assert history.it == []
    assert_states_equal(state, initial)


def test_invalid_step_state() -> None:
    sampler = Sampler(lambda x: np.zeros(len(x)), "pcn", dims=2)
    with pytest.raises(TypeError, match="StepState"):
        sampler.sample(np.zeros((3, 2)), 1, step_state=object(), verbose=False)


def no_target(x: Any) -> Any:
    pytest.fail("Backend validation must happen before target evaluation.")


def no_fit(*args: Any) -> Any:
    pytest.fail("Backend validation must happen before fitting.")


def test_rng_mismatch_does_not_advance_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("jax")
    x = np.random.default_rng(1).normal(size=(32, 2))
    sampler = Sampler(no_target, "pcn", dims=2, xp=np)
    rng = RandomGenerator(backend="jax", seed=21)
    expected_rng = RandomGenerator.from_state_dict(rng.state_dict())
    monkeypatch.setattr(PCNStep, "init_state", no_fit)
    with pytest.raises(ValueError, match="RNG backend.*sampler backend"):
        sampler.sample(x, 1, rng=rng, verbose=False)
    np.testing.assert_array_equal(
        to_numpy_array(rng.random((8,))),
        to_numpy_array(expected_rng.random((8,))),
    )


def test_implicit_rng_uses_sampler_backend(xp: Any) -> None:
    x = xp.asarray(np.random.default_rng(8).normal(size=(32, 2)))
    sampler = Sampler(
        lambda x: -0.5 * xp.sum(x**2, axis=-1),
        "pcn",
        dims=2,
        xp=xp,
    )
    first, _ = sampler.sample(x, 2, seed=4, verbose=False)
    second, _ = sampler.sample(x, 2, seed=4, verbose=False)
    np.testing.assert_array_equal(
        to_numpy_array(first), to_numpy_array(second)
    )
    assert array_namespace(first) is array_namespace(x)


def test_compatible_numpy_namespace() -> None:
    xp = np
    x = xp.asarray(np.random.default_rng(3).normal(size=(32, 2)))
    sampler = Sampler(
        lambda x: -0.5 * xp.sum(x**2, axis=-1),
        "pcn",
        dims=2,
        xp=array_namespace(x),
    )
    result, _ = sampler.sample(
        x,
        1,
        rng=np.random.default_rng(4),
        verbose=False,
        return_last_only=True,
    )
    assert array_namespace(result) is array_namespace(x)


def test_numpy_generator_mismatch_and_constructor_precedence() -> None:
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.asarray(np.random.default_rng(5).normal(size=(32, 2)))
    sampler = Sampler(no_target, "pcn", dims=2, xp=jnp)
    rng = np.random.default_rng(6)
    before = pickle.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="RNG backend 'numpy'"):
        sampler.sample(x, 1, rng=rng, verbose=False)
    assert pickle.dumps(rng.bit_generator.state) == before
    # A constructor RNG is validated too, after resolving call precedence.
    sampler.rng = rng
    with pytest.raises(ValueError, match="RNG backend 'numpy'"):
        sampler.sample(x, 1, verbose=False)
    sampler.log_prob_fn = lambda x: -0.5 * jnp.sum(x**2, axis=-1)
    with pytest.warns(UserWarning, match="Both"):
        sampler.sample(
            x, 1, rng=RandomGenerator(backend="jax", seed=7), verbose=False
        )


@pytest.mark.parametrize("functional", [False, True])
def test_array_mismatch_before_rng_resolution(
    functional: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    x = torch.asarray(np.random.default_rng(10).normal(size=(32, 2)))
    sampler = Sampler(no_target, "pcn", dims=2, xp=np)
    monkeypatch.setattr(PCNStep, "init_state", no_fit)

    def no_rng(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Array validation must happen before RNG resolution.")

    monkeypatch.setattr(sampler, "_resolve_rng", no_rng)
    with pytest.raises(
        ValueError, match="Input array backend.*sampler backend"
    ):
        if functional:
            sampler.sample_functional(x, 1, rng_state=None, verbose=False)
        else:
            sampler.sample(x, 1, verbose=False)


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize(
    "problem", ["step_type", "dimensions", "dtype", "backend"]
)
def test_incompatible_state_before_target(
    functional: bool, problem: str
) -> None:
    sampler = Sampler(no_target, "tpcn", dims=2)
    x = np.random.default_rng(17).normal(size=(32, 2))
    state = sampler.init_step_state(x)
    if problem == "step_type":
        state.nu = None
    elif problem == "dimensions":
        state.mu = state.mu[:1]
    elif problem == "dtype":
        x = x.astype(np.float32)
    else:
        jnp = pytest.importorskip("jax.numpy")
        state = sampler.init_step_state(jnp.asarray(x))
    with pytest.raises(ValueError, match="step_state"):
        if functional:
            sampler.sample_functional(
                x, 0, rng_state=None, step_state=state, verbose=False
            )
        else:
            sampler.sample(x, 0, seed=1, step_state=state, verbose=False)


def test_state_device_mismatch_before_target() -> None:
    torch = pytest.importorskip("torch")
    sampler = Sampler(no_target, "pcn", dims=2, xp=torch)
    x = torch.asarray(np.random.default_rng(17).normal(size=(32, 2)))
    state = sampler.init_step_state(x)
    # Exercise the device check without GPU hardware.
    state.rho = state.rho.to("meta")
    with pytest.raises(ValueError, match="step_state.rho.*backend"):
        sampler.sample(x, 0, seed=1, step_state=state, verbose=False)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_compatible_state_dtype_and_dimensions(
    xp: Any, dims: int, step_fn: str, dtype_name: str
) -> None:
    x = xp.asarray(
        np.random.default_rng(20).normal(size=(32, dims)),
        dtype=getattr(xp, dtype_name),
    )
    sampler = Sampler(
        lambda x: -0.5 * xp.sum(x**2, axis=-1), step_fn, dims, xp=xp
    )
    state = sampler.init_step_state(x)
    chain, _, final = sampler.sample(
        x, 1, seed=2, step_state=state, return_step_state=True, verbose=False
    )
    assert chain.dtype == x.dtype
    # Adaptation must leave the state compatible for the next segment.
    sampler.sample(chain[-1], 1, seed=3, step_state=final, verbose=False)


def test_import_without_acai():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['acai'] = None; import minipcn",
        ],
        check=True,
    )


@pytest.mark.parametrize("use_scan", [None, False, True])
def test_sampling_without_acai(monkeypatch, use_scan):
    monkeypatch.setitem(sys.modules, "acai", None)
    sampler = Sampler(lambda x: -0.5 * np.sum(x**2, axis=-1), "pcn", dims=2)
    x = np.random.default_rng(1).normal(size=(32, 2))
    if use_scan is True:
        with pytest.raises(
            ImportError, match=r"pip install 'minipcn\[scan\]'"
        ):
            sampler.sample(x, 2, seed=1, verbose=False, use_scan=True)
    else:
        chain, history = sampler.sample(
            x, 2, seed=1, verbose=False, use_scan=use_scan
        )
        assert chain.shape == (3, 32, 2)
        assert history.it == [0, 1]


@pytest.mark.parametrize("step", ["pcn", "tpcn"])
def test_jit_without_acai_matches_loop(monkeypatch, step):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    monkeypatch.setitem(sys.modules, "acai", None)
    sampler = Sampler(
        lambda x: -0.5 * jnp.sum(x**2, axis=-1), step, dims=2, xp=jnp
    )
    x = jnp.asarray(np.random.default_rng(1).normal(size=(32, 2)))
    key = jax.random.key(1)

    def run(x, key, use_scan):
        return sampler.sample_functional(
            x,
            2,
            rng_state=key,
            verbose=False,
            use_scan=use_scan,
            return_step_state=True,
        )

    actual = jax.jit(lambda x, key: run(x, key, None))(x, key)
    expected = jax.jit(lambda x, key: run(x, key, False))(x, key)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1].it, expected[1].it)
    np.testing.assert_array_equal(
        jax.random.key_data(actual[2]), jax.random.key_data(expected[2])
    )
    np.testing.assert_array_equal(actual[3].rho, expected[3].rho)
    assert actual[3].iteration == expected[3].iteration == 2


def test_disabled_scan_does_not_import_acai(monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "acai":
            pytest.fail("Disabled scan must not import acai")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    sampler = Sampler(lambda x: -0.5 * np.sum(x**2, axis=-1), "pcn", dims=2)
    x = np.random.default_rng(1).normal(size=(32, 2))
    sampler.sample(x, 2, seed=1, verbose=False, use_scan=False)
