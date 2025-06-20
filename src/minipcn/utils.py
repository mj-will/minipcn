from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import psi


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


def fit_student_t_em(x, nu_init=10.0, tol=1e-5, max_iter=1000):
    """Fit a multivariate Student's t-distribution using EM algorithm.

    Parameters
    ----------
    x : np.ndarray
        Samples of shape (n_samples, n_dims).
    nu_init : float, optional
        Initial degrees of freedom for the Student's t-distribution. Default is 10.0.
    tol : float, optional
        Tolerance for convergence of the degrees of freedom. Default is 1e-5.
    max_iter : int, optional
        Maximum number of iterations for the EM algorithm. Default is 1000.

    Returns
    -------
    mu : np.ndarray
        Mean of the fitted Student's t-distribution, shape (n_dims,).
    sigma : np.ndarray
        Covariance matrix of the fitted Student's t-distribution, shape (n_dims, n_dims).
    nu : float
        Estimated degrees of freedom of the Student's t-distribution.
    """
    n_samples, dims = x.shape
    t = 0
    mu = x.mean(axis=0)
    sigma = np.cov(x.T)
    nu = nu_init

    def mahalanobis(x, mu, Sigma_inv):
        diff = x - mu
        return diff @ Sigma_inv @ diff.T

    while t < max_iter:
        sigma_inv = np.linalg.inv(sigma)

        # Step 4: Compute δ_i^(t)
        delta = np.array([mahalanobis(xi, mu, sigma_inv) for xi in x])

        # Step 5: Compute w_i^(t+1)
        w = (nu + dims) / (nu + delta)

        # Step 6: Update μ^(t+1)
        mu_new = np.sum(w[:, None] * x, axis=0) / np.sum(w)

        # Update Σ^(t+1)
        diff = x - mu
        sigma_new = (
            w[:, None, None] * np.einsum("ni,nj->nij", diff, diff)
        ).sum(axis=0) / n_samples

        # Step 7: Recompute δ_i^(t+1)
        Sigma_new_inv = np.linalg.inv(sigma_new)
        delta_new = np.array(
            [mahalanobis(xi, mu_new, Sigma_new_inv) for xi in x]
        )

        # Step 8: Update w_i again for ν update
        w_i_nu = (nu + dims) / (nu + delta_new)

        # Step 9: Solve for ν using minimization (robust)
        def nu_equation(nu_val):
            term1 = -psi(nu_val / 2) + np.log(nu_val / 2)
            term2 = np.mean(np.log(w_i_nu) - w_i_nu)
            term3 = psi((nu_val + dims) / 2) - np.log((nu_val + dims) / 2)
            return term1 + term2 + term3

        res = minimize_scalar(
            lambda nu_val: (nu_equation(nu_val) + 1) ** 2,
            bounds=(1e-3, 1e6),
            method="bounded",
        )
        nu_new = res.x if res.success else nu  # fallback if minimization fails

        # Convergence check
        if abs(nu_new - nu) < tol:
            break

        # Update parameters
        mu, sigma, nu = mu_new, sigma_new, nu_new
        t += 1

    return mu, sigma, nu


def fit_gaussian(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a multivariate Gaussian to the samples.

    Parameters
    ----------
    x : np.ndarray
        Samples of shape (n_samples, n_dims).

    Returns
    -------
    mu : np.ndarray
        Mean of the fitted Gaussian, shape (n_dims,).
    cov : np.ndarray
        Covariance matrix of the fitted Gaussian, shape (n_dims, n_dims).
    """
    mu = x.mean(axis=0)
    cov = np.cov(x.T, bias=False)
    return mu, cov
