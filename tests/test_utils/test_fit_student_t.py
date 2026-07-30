import numpy as np
import pytest

from minipcn.student_t import fit_student_t_em
from minipcn.utils import to_numpy_array


@pytest.mark.parametrize("dims", [1, 2])
def test_fit_student_t_em_preserves_backend(xp, dims):
    samples = np.asarray(
        [
            [-2.0, 0.5],
            [-1.0, -0.5],
            [-0.25, 0.25],
            [0.0, 0.0],
            [0.25, -0.25],
            [1.0, 0.5],
            [8.0, -6.0],
        ],
        dtype=np.float32,
    )
    if dims == 1:
        samples = samples[:, 0]
    x = xp.asarray(samples, dtype=xp.float32)

    mu, shape, nu = fit_student_t_em(x, max_iter=100)

    assert mu.dtype == x.dtype
    assert shape.dtype == x.dtype
    assert nu.dtype == x.dtype
    assert mu.shape == (() if dims == 1 else (dims,))
    assert shape.shape == (() if dims == 1 else (dims, dims))
    assert nu.shape == ()
    assert np.all(np.isfinite(to_numpy_array(mu)))
    assert np.all(np.isfinite(to_numpy_array(shape)))
    assert np.isfinite(to_numpy_array(nu))
    assert float(to_numpy_array(nu)) > 0

    if dims == 1:
        assert float(to_numpy_array(shape)) > 0
    else:
        eigenvalues = np.linalg.eigvalsh(to_numpy_array(shape))
        assert np.all(eigenvalues > 0)


def test_fit_student_t_em_regularizes_constant_samples(xp):
    x = xp.asarray(np.ones((8, 2), dtype=np.float32), dtype=xp.float32)

    mu, shape, nu = fit_student_t_em(x, max_iter=20)

    np.testing.assert_allclose(to_numpy_array(mu), np.ones(2), atol=1e-5)
    assert np.all(np.linalg.eigvalsh(to_numpy_array(shape)) > 0)
    assert np.isfinite(to_numpy_array(nu))


def test_fit_student_t_em_jax_jit():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.asarray(
        [
            [-2.0, 0.5],
            [-1.0, -0.5],
            [0.0, 0.0],
            [1.0, 0.5],
            [8.0, -6.0],
        ],
        dtype=jnp.float32,
    )

    fit = jax.jit(lambda values: fit_student_t_em(values, max_iter=100))
    mu, shape, nu = fit(x)

    assert mu.shape == (2,)
    assert shape.shape == (2, 2)
    assert nu.shape == ()
    assert bool(jnp.all(jnp.isfinite(mu)))
    assert bool(jnp.all(jnp.isfinite(shape)))
    assert bool(jnp.isfinite(nu))
    assert bool(jnp.all(jnp.linalg.eigvalsh(shape) > 0))


@pytest.mark.parametrize(
    "samples",
    [
        np.asarray(1.0, dtype=np.float32),
        np.ones((2, 2, 2), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
    ],
)
def test_fit_student_t_em_rejects_invalid_sample_shapes(samples):
    with pytest.raises(ValueError):
        fit_student_t_em(samples)
