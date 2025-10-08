import numpy as np
import pytest


@pytest.fixture
def rng():
    """Fixture to provide a random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture(params=[1, 4])
def dims(request):
    return request.param


@pytest.fixture
def log_target_fn():
    """Fixture to provide a log target function."""

    def _log_target_fn(x):
        return -0.5 * np.sum(x**2, axis=-1)

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
