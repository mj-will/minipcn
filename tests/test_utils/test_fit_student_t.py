import numpy as np
import pytest
from scipy.special import polygamma, psi

from minipcn.student_t import _nu_value, _solve_nu, fit_student_t_em
from minipcn.utils import to_numpy_array


@pytest.mark.parametrize(
    ("dtype", "nu_init", "avg_term", "dims", "expected"),
    [
        (np.float64, 1e-3, -1.001, 1000, 618.3671231438226),
        (np.float32, 1e5, -1.1, 10, 6.492608031799467),
        (np.float32, 1e6, -1.8, 1000, 1.4953490086755392),
        (np.float32, 1e-6, -1.8, 1000, 1.4953490086755392),
        (np.float32, 1e9, -1.1, 10, 6.492608031799467),
    ],
)
def test_solve_nu_converges_from_distant_initial_value(
    dtype, nu_init, avg_term, dims, expected
):
    def digamma(value):
        return np.asarray(psi(value), dtype=dtype)

    def trigamma(value):
        return np.asarray(polygamma(1, value), dtype=dtype)

    nu = np.asarray(nu_init, dtype=dtype)
    avg = np.asarray(avg_term, dtype=dtype)
    dims_value = np.asarray(dims, dtype=dtype)

    result = _solve_nu(
        nu,
        avg,
        dims_value,
        xp=np,
        digamma=digamma,
        trigamma=trigamma,
    )
    residual = _nu_value(
        result,
        avg,
        dims_value,
        xp=np,
        digamma=digamma,
    )

    np.testing.assert_allclose(result, expected, rtol=5e-6)
    assert abs(float(residual)) <= max(1e-8, np.finfo(dtype).eps)


@pytest.mark.parametrize(
    ("nu_init", "avg_term", "dims", "expected"),
    [
        (1e5, -1.1, 10, 6.492608031799467),
        (1e6, -1.8, 1000, 1.4953490086755392),
        (1e-6, -1.8, 1000, 1.4953490086755392),
        (1e9, -1.1, 10, 6.492608031799467),
    ],
)
def test_solve_nu_jax_converges(
    nu_init,
    avg_term,
    dims,
    expected,
):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jsp = pytest.importorskip("jax.scipy.special")
    from minipcn._jax import _solve_nu as solve_nu_jax

    dtype = jnp.float32
    nu = jnp.asarray(nu_init, dtype=dtype)
    avg = jnp.asarray(avg_term, dtype=dtype)
    dims_value = jnp.asarray(dims, dtype=dtype)

    result = jax.jit(solve_nu_jax)(nu, avg, dims_value)
    residual = _nu_value(
        result,
        avg,
        dims_value,
        xp=jnp,
        digamma=jsp.digamma,
    )

    np.testing.assert_allclose(result, expected, rtol=5e-6)
    assert abs(float(residual)) <= max(1e-8, np.finfo(np.float32).eps)


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
