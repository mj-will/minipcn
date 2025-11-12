import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np

try:
    from array_api_compat import array_namespace
    from array_api_compat import device as get_device
except ImportError:
    logging.warning(
        "array_api_compat is not installed. Falling back to numpy for array namespace."
    )

    def array_namespace(x: Any) -> Any:
        import numpy as np

        return np

    def get_device(x: Any) -> None:
        return None


from scipy.linalg import solve_triangular
from scipy.optimize import root_scalar
from scipy.special import polygamma, psi

from ._typing import Array


@dataclass
class ChainState:
    """State of the chain at a given iteration.

    Attributes
    ----------
    it : int
        Current iteration number.
    acceptance_rate : float
        Acceptance rate of the current iteration.
    target_acceptance_rate : float
        Target acceptance rate for the chain.
    step : str
        Name of the step function used in this iteration.
    extra_stats : Dict[str, Any]
        Additional statistics collected during the iteration.
    """

    it: int
    acceptance_rate: float
    target_acceptance_rate: float
    step: str = ""
    extra_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainStateHistory:
    it: List[int]
    acceptance_rate: List[float]
    target_acceptance_rate: List[float]
    step: str = ""
    extra_stats: Dict[str, List[Any]] = field(default_factory=dict)

    def __getitem__(self, index: Union[int, slice]) -> "ChainStateHistory":
        # Support slicing or single index
        if isinstance(index, int):
            return ChainStateHistory(
                it=[self.it[index]],
                acceptance_rate=[self.acceptance_rate[index]],
                target_acceptance_rate=[self.target_acceptance_rate[index]],
                step=self.step,
                extra_stats={
                    k: [v[index]] for k, v in self.extra_stats.items()
                },
            )
        elif isinstance(index, slice):
            return ChainStateHistory(
                it=self.it[index],
                acceptance_rate=self.acceptance_rate[index],
                target_acceptance_rate=self.target_acceptance_rate[index],
                step=self.step,
                extra_stats={k: v[index] for k, v in self.extra_stats.items()},
            )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    @classmethod
    def from_chain_states(
        cls, states: List[ChainState]
    ) -> "ChainStateHistory":
        extra_stats = {}
        for key in states[0].extra_stats.keys():
            extra_stats[key] = [s.extra_stats[key] for s in states]
        return cls(
            it=[s.it for s in states],
            acceptance_rate=[s.acceptance_rate for s in states],
            target_acceptance_rate=[s.target_acceptance_rate for s in states],
            extra_stats=extra_stats,
        )

    def plot_acceptance_rate(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.it, self.acceptance_rate, label="Acceptance Rate")
        ax.plot(
            self.it,
            self.target_acceptance_rate,
            label="Target Acceptance Rate",
            linestyle="--",
            color="k",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Acceptance Rate")
        ax.legend()
        return fig

    def plot_extra_stat(self, key: str):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.it, self.extra_stats[key], label=key)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(key)
        ax.legend()
        return fig


def to_numpy_array(x: Array) -> np.ndarray:
    """Convert an array-like object to a NumPy array.

    Handles special cases for PyTorch and CuPy arrays.

    Parameters
    ----------
    x : Array
        Input array-like object.

    Returns
    -------
    np.ndarray
        Converted NumPy array.
    """
    try:
        return np.asarray(x)
    except Exception:
        from array_api_compat import is_cupy_array, is_torch_array

        if is_torch_array(x):
            return np.asarray(x.detach().cpu())
        elif is_cupy_array(x):
            return np.asarray(x.get())
        else:
            raise


def fit_student_t_em(
    x: Array,
    nu_init: float = 10.0,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> tuple[Array, Array, Array]:
    """Fit a multivariate Student's t-distribution using EM algorithm.

    Parameters
    ----------
    x : Array
        Samples of shape (n_samples, n_dims).
    nu_init : float, optional
        Initial degrees of freedom for the Student's t-distribution. Default is 10.0.
    tol : float, optional
        Tolerance for convergence of the degrees of freedom. Default is 1e-5.
    max_iter : int, optional
        Maximum number of iterations for the EM algorithm. Default is 1000.

    Returns
    -------
    mu : Array
        Mean of the fitted Student's t-distribution, shape (n_dims,).
    sigma : Array
        Covariance matrix of the fitted Student's t-distribution, shape (n_dims, n_dims).
    nu : float
        Estimated degrees of freedom of the Student's t-distribution.
    """
    # Ensure x is 2D

    xp = array_namespace(x)
    dtype = x.dtype

    device = get_device(x)

    x = to_numpy_array(x)

    x = np.atleast_2d(x)
    if x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T
    n_samples, dims = x.shape

    mu = x.mean(axis=0)
    sigma = np.cov(x.T) if dims > 1 else float(np.var(x, ddof=1))
    nu = nu_init

    min_variance = 1e-9

    def ensure_positive_definite(matrix):
        if dims == 1:
            # Guarantee strictly positive variance to avoid divide-by-zero.
            adjusted = float(max(matrix, min_variance))
            return adjusted, None

        adjusted = np.array(matrix, copy=True)
        eye = np.eye(dims)
        scale = np.mean(np.diag(adjusted))
        if scale <= 0:
            scale = 1.0
        jitter = 0.0
        for _ in range(8):
            try:
                chol = np.linalg.cholesky(adjusted)
                return adjusted, chol
            except np.linalg.LinAlgError:
                jitter = max(
                    min_variance, (1e-9 if jitter == 0.0 else jitter * 10.0)
                )
                adjusted = adjusted + eye * (jitter * scale)
        raise np.linalg.LinAlgError(
            "Matrix is not positive definite even after jitter"
        )

    def mahalanobis_squared(diff_mat, chol):
        solved = solve_triangular(
            chol,
            diff_mat.T,
            lower=True,
            check_finite=False,
        ).T
        return np.sum(solved**2, axis=1)

    for _ in range(max_iter):
        diff = x - mu
        if dims == 1:
            sigma, _ = ensure_positive_definite(sigma)
            delta = (diff[:, 0] ** 2) / sigma
        else:
            sigma, chol = ensure_positive_definite(sigma)
            delta = mahalanobis_squared(diff, chol)

        w = (nu + dims) / (nu + delta)
        w_sum = np.sum(w)
        mu_new = (w[:, None] * x).sum(axis=0) / w_sum

        diff_new = x - mu_new
        if dims == 1:
            sigma_new = float(np.dot(w, diff_new[:, 0] ** 2) / w_sum)
            sigma_new, _ = ensure_positive_definite(sigma_new)
            delta_new = (diff_new[:, 0] ** 2) / sigma_new
        else:
            sigma_new = (diff_new.T * w) @ diff_new / w_sum
            sigma_new = 0.5 * (sigma_new + sigma_new.T)
            sigma_new, chol_new = ensure_positive_definite(sigma_new)
            delta_new = mahalanobis_squared(diff_new, chol_new)

        w_i_nu = (nu + dims) / (nu + delta_new)
        avg_log_w_minus_w = np.mean(np.log(w_i_nu) - w_i_nu)

        def nu_equation(nu_val):
            return (
                -psi(nu_val / 2.0)
                + np.log(nu_val / 2.0)
                + 1.0
                + avg_log_w_minus_w
                + psi((nu_val + dims) / 2.0)
                - np.log((nu_val + dims) / 2.0)
            )

        def nu_equation_prime(nu_val):
            return (
                -0.5 * polygamma(1, nu_val / 2.0)
                + 1.0 / nu_val
                + 0.5 * polygamma(1, (nu_val + dims) / 2.0)
                - 1.0 / (nu_val + dims)
            )

        nu_new = nu
        try:
            root_res = root_scalar(
                nu_equation,
                fprime=nu_equation_prime,
                x0=nu,
                method="newton",
            )
            if root_res.converged and root_res.root > 0:
                nu_new = root_res.root
        except (ValueError, RuntimeError):
            # Fall back to the previous value if Newton iteration fails.
            pass

        mu_diff = float(np.max(np.abs(mu_new - mu)))
        sigma_diff = (
            abs(sigma_new - sigma)
            if dims == 1
            else float(np.max(np.abs(sigma_new - sigma)))
        )

        if max(mu_diff, sigma_diff, abs(nu_new - nu)) < tol:
            mu, sigma, nu = mu_new, sigma_new, nu_new
            break

        mu, sigma, nu = mu_new, sigma_new, nu_new

    # Return scalar for 1D
    if dims == 1:
        mu = mu.item()
        sigma = float(sigma)

    if device is not None:
        mu = xp.asarray(mu, dtype=dtype, device=device)
        sigma = xp.asarray(sigma, dtype=dtype, device=device)
        nu = xp.asarray(nu, dtype=dtype, device=device)
    else:
        mu = xp.asarray(mu, dtype=dtype)
        sigma = xp.asarray(sigma, dtype=dtype)
        nu = xp.asarray(nu, dtype=dtype)

    return mu, sigma, nu


def fit_gaussian(x: Array) -> tuple[Array, Array]:
    """
    Fit a multivariate Gaussian to the samples.

    Parameters
    ----------
    x : Array
        Samples of shape (n_samples, n_dims).

    Returns
    -------
    mu : Array
        Mean of the fitted Gaussian, shape (n_dims,).
    cov : Array
        Covariance matrix of the fitted Gaussian, shape (n_dims, n_dims).
    """
    xp = array_namespace(x)
    mu = xp.mean(x, axis=0)
    cov = xp.cov(x.T)
    return mu, cov
