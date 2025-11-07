from typing import Any

import numpy as np

from ._typing import Array
from .utils import ChainState


class Step:
    """Base class for a step in the MiniPCN sampler.

    Parameters
    ----------
    dims : int
        Number of dimensions of the target distribution.
    rng : np.random.Generator
        Random number generator.
    """

    def __init__(self, dims: int, rng: np.random.Generator, xp: Any):
        self.dims = dims
        self.rng = rng
        self.xp = xp

    def initialise(self, x: Array):
        pass

    def update(self, state: ChainState, samples: Array):
        pass

    def update_state(self, state: ChainState) -> ChainState:
        state.step = self.__class__.__name__
        return state

    def step(self, x: Array) -> tuple[Array, Array]:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


class PCNStep(Step):
    """Preconditioned Crank-Nicolson step.

    This uses the standard pCN proposal.

    Parameters
    ----------
    dims : int
        Number of dimensions of the target distribution.
    rng : np.random.Generator
        Random number generator.
    rho : float, optional
        pCN step size parameter, must be in the range (0, 1). Default is 0.5.
        See https://arxiv.org/abs/2407.07781 for details.
    """

    def __init__(self, dims, rng, xp, rho: float = 0.5):
        super().__init__(dims, rng, xp)

        if not (0 < rho < 1):
            raise ValueError("rho must be in the range (0, 1).")
        self.rho = rho

    def initialise(self, x):
        from .utils import fit_gaussian

        self.mu, self.cov = fit_gaussian(x)
        if self.dims == 1:
            self.inv_cov = self.xp.atleast_2d(1.0 / self.cov)
            self.chol_cov = self.xp.atleast_2d(self.xp.sqrt(self.cov))
        else:
            self.inv_cov = self.xp.linalg.inv(self.cov)
            self.chol_cov = self.xp.linalg.cholesky(self.cov)

    def update(self, state, samples):
        delta = state.acceptance_rate - state.target_acceptance_rate
        step_size = 1 / (state.it + 1) ** 0.75
        self.rho = np.abs(
            np.minimum(
                self.rho + step_size * delta,
                np.minimum(2.38 / self.dims**0.5, 0.99),
            )
        )

    def update_state(self, state):
        state = super().update_state(state)
        state.extra_stats["rho"] = self.rho
        return state

    def step(self, x):
        n_samples = x.shape[0]
        diff = x - self.mu  # (N, D)

        # Sample W_m ~ N(0, C)
        z = self.rng.normal(size=(n_samples, self.dims))
        w = (self.chol_cov @ z.T).T  # (N, D)

        # Proposed new samples x'
        x_prime = (
            self.mu
            + self.xp.sqrt(self.xp.asarray(1 - self.rho**2)) * diff
            + self.rho * w
        )

        # Evaluate the log proposal density:
        # Since C is constant, we can ignore normalizing terms for computing alpha.

        diff_prime = x_prime - self.mu

        # Mahalanobis distances
        m_x = self.xp.einsum("ni,ij,nj->n", diff, self.inv_cov, diff)
        m_xp = self.xp.einsum(
            "ni,ij,nj->n", diff_prime, self.inv_cov, diff_prime
        )

        log_alpha = -0.5 * (m_x - m_xp)

        return x_prime, log_alpha


class TPCNStep(PCNStep):
    """t-preconditioned Crank-Nicolson step.

    This uses a Student-t distribution for the proposal. See
    https://arxiv.org/abs/2407.07781 for details.

    Parameters
    ----------
    dims : int
        Number of dimensions of the target distribution.
    rng : np.random.Generator
        Random number generator.
    rho : float, optional
        pCN step size parameter, must be in the range (0, 1). Default is 0.5.
        See https://arxiv.org/abs/2407.07781 for details.
    """

    def initialise(self, x):
        from .utils import fit_student_t_em

        self.mu, self.cov, self.nu = fit_student_t_em(x)
        if self.dims == 1:
            self.inv_cov = self.xp.atleast_2d(1.0 / self.cov)
            self.chol_cov = self.xp.atleast_2d(self.xp.sqrt(self.cov))
        else:
            self.inv_cov = self.xp.linalg.inv(self.cov)
            self.chol_cov = self.xp.linalg.cholesky(self.cov)

    def step(self, x):
        n_samples = x.shape[0]
        diff = x - self.mu  # Shape: (N, D)

        # Mahalanobis distances
        xx = self.xp.einsum(
            "ni,ij,nj->n", diff, self.inv_cov, diff
        )  # Shape: (N,)
        k = 0.5 * (self.dims + self.nu)
        theta = 2 / (self.nu + xx)
        z_inv = 1 / self.rng.gamma(shape=k, scale=theta)  # Shape: (N,)

        # Propose new samples
        z = self.rng.normal(size=(n_samples, self.dims))
        scaled_noise = (
            (self.xp.sqrt(z_inv)[:, None]) * (self.chol_cov @ z.T).T
        )  # Shape: (N, D)
        x_prime = (
            self.mu
            + self.xp.sqrt(self.xp.asarray(1 - self.rho**2)) * diff
            + self.rho * scaled_noise
        )  # Shape: (N, D)

        diff_prime = x_prime - self.mu
        xx_prime = self.xp.einsum(
            "ni,ij,nj->n", diff_prime, self.inv_cov, diff_prime
        )

        log_a_num = (-0.5 * (self.nu + self.dims)) * self.xp.log1p(
            xx / self.nu
        )
        log_a_denom = (-0.5 * (self.nu + self.dims)) * self.xp.log1p(
            xx_prime / self.nu
        )
        log_alpha = log_a_num - log_a_denom
        return x_prime, log_alpha


def step_factory(
    step_name: str, dims: int, rng: np.random.Generator, xp, **kwargs
):
    """
    Factory function to create a step instance based on the step name.

    Parameters
    ----------
    step_name : {"pCN", "tPCN"}
        Name of the step type.
    dims : int
        Number of dimensions of the target distribution.
    rng : np.random.Generator
        Random number generator.
    **kwargs : dict
        Additional keyword arguments to pass to the step constructor.
    """
    if step_name.lower() == "pcn":
        return PCNStep(dims=dims, rng=rng, xp=xp, **kwargs)
    elif step_name.lower() == "tpcn":
        return TPCNStep(dims=dims, rng=rng, xp=xp, **kwargs)
    else:
        raise ValueError(f"Unknown step type: {step_name}")
