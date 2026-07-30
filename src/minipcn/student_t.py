from typing import Any

from array_api_compat import array_namespace, is_jax_array, is_torch_array
from array_api_compat import device as get_device
from scipy.special import polygamma, psi

from ._typing import Array


def _prepare_samples(x: Array, xp: Any) -> Array:
    if x.ndim == 1:
        x = xp.expand_dims(x, axis=1)
    elif x.ndim != 2:
        raise ValueError("x must be a one- or two-dimensional array")

    if x.shape[0] == 1 and x.shape[1] > 1:
        x = xp.permute_dims(x, (1, 0))
    if x.shape[0] < 2:
        raise ValueError("x must contain at least two samples")
    return x


def _nu_value(nu, avg_term, dims, *, xp, digamma):
    return (
        -digamma(nu / 2)
        + xp.log(nu / 2)
        + 1
        + avg_term
        + digamma((nu + dims) / 2)
        - xp.log((nu + dims) / 2)
    )


def _solve_nu(
    nu,
    avg_term,
    dims,
    *,
    xp,
    digamma,
    trigamma,
    max_iter: int = 12,
):
    """Solve the Student-t degrees-of-freedom equation in log-space."""
    dtype = nu.dtype
    eps = xp.asarray(xp.finfo(dtype).eps, dtype=dtype)
    eta_min = xp.log(xp.asarray(1e-3, dtype=dtype))
    eta_max = xp.log(xp.asarray(1e6, dtype=dtype))
    eta = xp.log(nu)
    initial_residual = xp.abs(
        _nu_value(
            nu,
            avg_term,
            dims,
            xp=xp,
            digamma=digamma,
        )
    )

    for _ in range(max_iter):
        nu_current = xp.exp(eta)
        value = _nu_value(
            nu_current,
            avg_term,
            dims,
            xp=xp,
            digamma=digamma,
        )
        derivative = (
            -0.5 * trigamma(nu_current / 2)
            + 1 / nu_current
            + 0.5 * trigamma((nu_current + dims) / 2)
            - 1 / (nu_current + dims)
        )
        denominator = nu_current * derivative
        safe = xp.logical_and(
            xp.logical_and(xp.isfinite(value), xp.isfinite(denominator)),
            xp.abs(denominator) > eps,
        )
        step = xp.where(safe, value / denominator, xp.zeros_like(value))
        step = xp.clip(step, -2.0, 2.0)
        candidate = xp.clip(eta - step, eta_min, eta_max)
        eta = xp.where(xp.isfinite(candidate), candidate, eta)

    candidate = xp.exp(eta)
    final_residual = xp.abs(
        _nu_value(
            candidate,
            avg_term,
            dims,
            xp=xp,
            digamma=digamma,
        )
    )
    improved = xp.logical_and(
        xp.isfinite(final_residual),
        final_residual <= initial_residual,
    )
    return xp.where(improved, candidate, nu)


def _fit_eager(
    x: Array,
    nu_init: float,
    tol: float,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    xp = array_namespace(x)
    x = _prepare_samples(x, xp)
    n_samples, dims = x.shape
    dtype = x.dtype
    device = get_device(x)
    asarray_kwargs = {"dtype": dtype}
    if device is not None:
        asarray_kwargs["device"] = device

    if is_torch_array(x):
        import torch

        digamma = torch.special.digamma

        def trigamma(value):
            return torch.special.polygamma(1, value)

    else:

        def digamma(value):
            return xp.asarray(psi(value), **asarray_kwargs)

        def trigamma(value):
            return xp.asarray(polygamma(1, value), **asarray_kwargs)

    dims_value = xp.asarray(dims, **asarray_kwargs)
    one = xp.asarray(1, **asarray_kwargs)
    eye = xp.eye(dims, **asarray_kwargs)
    ridge_factor = xp.asarray(
        max(1e-9, 10 * xp.finfo(dtype).eps),
        **asarray_kwargs,
    )

    def regularize(matrix):
        matrix = 0.5 * (matrix + xp.permute_dims(matrix, (1, 0)))
        scale = xp.maximum(xp.mean(xp.linalg.diagonal(matrix)), one)
        return matrix + ridge_factor * scale * eye

    def mahalanobis_squared(diff, sigma):
        chol = xp.linalg.cholesky(sigma)
        solved = xp.linalg.solve(
            chol,
            xp.permute_dims(diff, (1, 0)),
        )
        return xp.sum(solved**2, axis=0)

    mu = xp.mean(x, axis=0)
    centered = x - mu
    sigma = (xp.permute_dims(centered, (1, 0)) @ centered) / xp.asarray(
        n_samples - 1, **asarray_kwargs
    )
    sigma = regularize(sigma)
    nu = xp.asarray(nu_init, **asarray_kwargs)

    for _ in range(max_iter):
        diff = x - mu
        delta = mahalanobis_squared(diff, sigma)
        weights = (nu + dims_value) / (nu + delta)
        weight_sum = xp.sum(weights)
        mu_new = xp.sum(weights[:, None] * x, axis=0) / weight_sum

        diff_new = x - mu_new
        sigma_new = (
            xp.permute_dims(diff_new, (1, 0)) @ (weights[:, None] * diff_new)
        ) / weight_sum
        sigma_new = regularize(sigma_new)
        delta_new = mahalanobis_squared(diff_new, sigma_new)

        nu_weights = (nu + dims_value) / (nu + delta_new)
        avg_term = xp.mean(xp.log(nu_weights) - nu_weights)
        nu_new = _solve_nu(
            nu,
            avg_term,
            dims_value,
            xp=xp,
            digamma=digamma,
            trigamma=trigamma,
        )
        error = xp.maximum(
            xp.max(xp.abs(mu_new - mu)),
            xp.maximum(
                xp.max(xp.abs(sigma_new - sigma)),
                xp.abs(nu_new - nu),
            ),
        )
        mu, sigma, nu = mu_new, sigma_new, nu_new
        if float(error) < tol:
            break

    if dims == 1:
        return mu[0], sigma[0, 0], nu
    return mu, sigma, nu


def fit_student_t_em(
    x: Array,
    nu_init: float = 10.0,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> tuple[Array, Array, Array]:
    """Fit a multivariate Student's t-distribution using EM.

    Parameters
    ----------
    x : Array
        Samples of shape ``(n_samples, n_dims)``.
    nu_init : float, optional
        Initial degrees of freedom. Default is 10.
    tol : float, optional
        Convergence tolerance. Default is 1e-5.
    max_iter : int, optional
        Maximum number of EM iterations. Default is 1000.

    Returns
    -------
    mu : Array
        Fitted location.
    shape : Array
        Fitted Student-t shape matrix.
    nu : Array
        Fitted degrees of freedom.
    """
    if is_jax_array(x):
        from ._jax import fit_student_t_em as fit_student_t_em_jax

        return fit_student_t_em_jax(x, nu_init, tol, max_iter)
    return _fit_eager(x, nu_init, tol, max_iter)
