import numpy as np
import pytest


@pytest.fixture(params=["numpy", "jax", "torch"])
def backend(request):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture
def rng(backend):
    """Fixture to provide a random number generator."""
    try:
        from orng import ArrayRNG

        return ArrayRNG(backend=backend, seed=42)
    except ImportError:
        return np.random.default_rng(seed=42)


@pytest.fixture(params=[1, 4])
def dims(request):
    return request.param


@pytest.fixture()
def xp(backend):
    """Fixture to provide the array namespace based on the backend."""
    if backend == "numpy":
        import numpy as xp
    elif backend == "jax":
        import jax.numpy as xp
    elif backend == "torch":
        import torch as xp
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return xp


@pytest.fixture
def log_target_fn(xp):
    """Fixture to provide a log target function."""

    def _log_target_fn(x):
        return -0.5 * xp.sum(x**2, -1)

    return _log_target_fn


@pytest.fixture(params=["tpCN", "pCN"])
def step_fn(request):
    return request.param


@pytest.fixture
def close_figures():
    """Fixture to close all matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
