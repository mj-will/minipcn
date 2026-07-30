from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from array_api_compat import device as get_device

from ._typing import Array
from .student_t import _nu_value, _prepare_samples


def _is_tracer(value: Any) -> bool:
    return isinstance(value, jax.core.Tracer)


def _contains_tracer(value: Any) -> bool:
    if _is_tracer(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tracer(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tracer(item) for item in value)
    return False


def _stack_history_values(values: list[Any], xp: Any) -> Any:
    return xp.stack([xp.asarray(value) for value in values], axis=0)


def _register_dataclass(*args, **kwargs):
    return jax.tree_util.register_dataclass(*args, **kwargs)


def fit_student_t_em(
    x: Array,
    nu_init: float,
    tol: float,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    """Fit a Student-t distribution using JAX-compatible compiled loops."""
    x = _prepare_samples(x, jnp)
    n_samples, dims = x.shape
    dtype = x.dtype
    device = get_device(x)
    asarray_kwargs: dict[str, Any] = {"dtype": dtype}
    if device is not None:
        asarray_kwargs["device"] = device

    dims_value = jnp.asarray(dims, **asarray_kwargs)
    one = jnp.asarray(1, **asarray_kwargs)
    tol_value = jnp.asarray(tol, **asarray_kwargs)
    eye = jnp.eye(dims, **asarray_kwargs)
    ridge_factor = jnp.asarray(
        max(1e-9, 10 * jnp.finfo(dtype).eps),
        **asarray_kwargs,
    )

    def regularize(matrix):
        matrix = 0.5 * (matrix + jnp.permute_dims(matrix, (1, 0)))
        scale = jnp.maximum(jnp.mean(jnp.diag(matrix)), one)
        return matrix + ridge_factor * scale * eye

    def mahalanobis_squared(diff, sigma):
        chol = jnp.linalg.cholesky(sigma)
        solved = jnp.linalg.solve(
            chol,
            jnp.permute_dims(diff, (1, 0)),
        )
        return jnp.sum(solved**2, axis=0)

    def solve_nu(nu, avg_term):
        eps = jnp.asarray(jnp.finfo(dtype).eps, **asarray_kwargs)
        eta_min = jnp.log(jnp.asarray(1e-3, **asarray_kwargs))
        eta_max = jnp.log(jnp.asarray(1e6, **asarray_kwargs))

        def body(_, eta):
            nu_current = jnp.exp(eta)
            value = _nu_value(
                nu_current,
                avg_term,
                dims_value,
                xp=jnp,
                digamma=jsp.digamma,
            )
            derivative = (
                -0.5 * jsp.polygamma(1, nu_current / 2)
                + 1 / nu_current
                + 0.5 * jsp.polygamma(1, (nu_current + dims_value) / 2)
                - 1 / (nu_current + dims_value)
            )
            denominator = nu_current * derivative
            safe = (
                jnp.isfinite(value)
                & jnp.isfinite(denominator)
                & (jnp.abs(denominator) > eps)
            )
            step = jnp.where(safe, value / denominator, 0)
            step = jnp.clip(step, -2.0, 2.0)
            candidate = jnp.clip(eta - step, eta_min, eta_max)
            return jnp.where(jnp.isfinite(candidate), candidate, eta)

        initial_residual = jnp.abs(
            _nu_value(
                nu,
                avg_term,
                dims_value,
                xp=jnp,
                digamma=jsp.digamma,
            )
        )
        eta = jax.lax.fori_loop(0, 12, body, jnp.log(nu))
        candidate = jnp.exp(eta)
        final_residual = jnp.abs(
            _nu_value(
                candidate,
                avg_term,
                dims_value,
                xp=jnp,
                digamma=jsp.digamma,
            )
        )
        improved = jnp.isfinite(final_residual) & (
            final_residual <= initial_residual
        )
        return jnp.where(improved, candidate, nu)

    mu = jnp.mean(x, axis=0)
    centered = x - mu
    sigma = (jnp.permute_dims(centered, (1, 0)) @ centered) / jnp.asarray(
        n_samples - 1, **asarray_kwargs
    )
    sigma = regularize(sigma)
    nu = jnp.asarray(nu_init, **asarray_kwargs)

    def condition(state):
        iteration, _, _, _, error = state
        return (iteration < max_iter) & (error > tol_value)

    def body(state):
        iteration, mu, sigma, nu, _ = state
        diff = x - mu
        delta = mahalanobis_squared(diff, sigma)
        weights = (nu + dims_value) / (nu + delta)
        weight_sum = jnp.sum(weights)
        mu_new = jnp.sum(weights[:, None] * x, axis=0) / weight_sum

        diff_new = x - mu_new
        sigma_new = (
            jnp.permute_dims(diff_new, (1, 0)) @ (weights[:, None] * diff_new)
        ) / weight_sum
        sigma_new = regularize(sigma_new)
        delta_new = mahalanobis_squared(diff_new, sigma_new)

        nu_weights = (nu + dims_value) / (nu + delta_new)
        avg_term = jnp.mean(jnp.log(nu_weights) - nu_weights)
        nu_new = solve_nu(nu, avg_term)

        error = jnp.maximum(
            jnp.max(jnp.abs(mu_new - mu)),
            jnp.maximum(
                jnp.max(jnp.abs(sigma_new - sigma)),
                jnp.abs(nu_new - nu),
            ),
        )
        valid = (
            jnp.all(jnp.isfinite(mu_new))
            & jnp.all(jnp.isfinite(sigma_new))
            & jnp.isfinite(nu_new)
            & jnp.isfinite(error)
        )
        return (
            iteration + 1,
            jnp.where(valid, mu_new, mu),
            jnp.where(valid, sigma_new, sigma),
            jnp.where(valid, nu_new, nu),
            jnp.where(valid, error, 0),
        )

    initial_state = (
        jnp.asarray(0),
        mu,
        sigma,
        nu,
        jnp.asarray(jnp.inf, **asarray_kwargs),
    )
    _, mu, sigma, nu, _ = jax.lax.while_loop(
        condition,
        body,
        initial_state,
    )

    if dims == 1:
        return mu[0], sigma[0, 0], nu
    return mu, sigma, nu
