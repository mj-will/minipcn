from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orng.functional import FunctionalBackend

from ._typing import Array
from .utils import ChainState, register_dataclass


@register_dataclass
@dataclass
class StepState:
    mu: Any
    cov: Any
    inv_cov: Any
    chol_cov: Any
    rho: Any
    nu: Any | None = None


class Step:
    """Base class for a step in the MiniPCN sampler.

    Parameters
    ----------
    dims : int
        The dimensionality of the samples.
    xp : Any
        The array library to use (e.g., numpy, jax.numpy).
    rng_backend : orng.functional.FunctionalBackend
        The random number generator backend to use.
    """

    def __init__(self, dims: int, xp: Any, rng_backend: FunctionalBackend):
        self.dims = dims
        self.xp = xp
        self.rng_backend = rng_backend

    def init_state(self, x: Array) -> StepState:
        raise NotImplementedError("Subclasses should implement this method.")

    def propose(
        self,
        state: StepState,
        rng_state: Any,
        x: Array,
    ) -> tuple[Any, Array, Array]:
        raise NotImplementedError("Subclasses should implement this method.")

    def adapt(
        self,
        state: StepState,
        chain_state: ChainState,
        *,
        samples: Array,
    ) -> tuple[StepState, ChainState]:
        return state, chain_state

    @property
    def step_name(self) -> str:
        return self.__class__.__name__


class PCNStep(Step):
    """Preconditioned Crank-Nicolson step.

    This uses the standard pCN proposal.

    Parameters
    ----------
    dims : int
        Number of dimensions of the target distribution.
    xp : Any
        The array library to use (e.g., numpy, jax.numpy).
    rng_backend : orng.functional.FunctionalBackend
        The random number generator backend to use.
    rho : float, optional
        pCN step size parameter, must be in the range (0, 1). Default is 0.5.
        See https://arxiv.org/abs/2407.07781 for details.
    adaptive : bool, optional
        Whether to adapt the rho parameter during sampling. Default is True.
    """

    def __init__(
        self,
        dims: int,
        xp: Any,
        rng_backend: Any,
        rho: float = 0.5,
        adaptive: bool = True,
    ):
        super().__init__(dims, xp, rng_backend)
        if not (0 < rho < 1):
            raise ValueError("rho must be in the range (0, 1).")
        self.rho = rho
        self.adaptive = adaptive

    def init_state(self, x: Array) -> StepState:
        from .utils import fit_gaussian

        mu, cov = fit_gaussian(x)
        if self.dims == 1:
            inv_cov = self.xp.atleast_2d(1.0 / cov)
            chol_cov = self.xp.atleast_2d(self.xp.sqrt(cov))
        else:
            inv_cov = self.xp.linalg.inv(cov)
            chol_cov = self.xp.linalg.cholesky(cov)
        rho = self.xp.asarray(self.rho, dtype=x.dtype)
        return StepState(
            mu=mu,
            cov=cov,
            inv_cov=inv_cov,
            chol_cov=chol_cov,
            rho=rho,
        )

    def propose(
        self,
        state: StepState,
        rng_state: Any,
        x: Array,
    ) -> tuple[Any, Array, Array]:
        n_samples = x.shape[0]
        diff = x - state.mu

        z, rng_state = self.rng_backend.normal(
            rng_state,
            loc=0.0,
            scale=1.0,
            size=(n_samples, self.dims),
            dtype=x.dtype,
        )
        w = (state.chol_cov @ z.T).T
        rho = state.rho
        x_prime = (
            state.mu
            + self.xp.sqrt(self.xp.asarray(1.0, dtype=x.dtype) - rho**2) * diff
            + rho * w
        )

        diff_prime = x_prime - state.mu
        m_x = self.xp.einsum("ni,ij,nj->n", diff, state.inv_cov, diff)
        m_xp = self.xp.einsum(
            "ni,ij,nj->n", diff_prime, state.inv_cov, diff_prime
        )
        log_alpha = -0.5 * (m_x - m_xp)
        return rng_state, x_prime, log_alpha

    def adapt(
        self,
        state: StepState,
        chain_state: ChainState,
        *,
        samples: Array,
    ) -> tuple[StepState, ChainState]:
        del samples
        if not self.adaptive:
            return state, chain_state
        dtype = state.rho.dtype
        step_size = self.xp.asarray(chain_state.it + 1, dtype=dtype) ** (-0.75)
        rho_next = self.xp.abs(
            self.xp.minimum(
                state.rho
                + step_size
                * (
                    chain_state.acceptance_rate
                    - self.xp.asarray(
                        chain_state.target_acceptance_rate, dtype=dtype
                    )
                ),
                self.xp.minimum(
                    self.xp.asarray(2.38 / self.dims**0.5, dtype=dtype),
                    self.xp.asarray(0.99, dtype=dtype),
                ),
            )
        )
        next_state = StepState(
            mu=state.mu,
            cov=state.cov,
            inv_cov=state.inv_cov,
            chol_cov=state.chol_cov,
            rho=rho_next,
            nu=state.nu,
        )
        next_chain_state = ChainState(
            it=chain_state.it,
            acceptance_rate=chain_state.acceptance_rate,
            target_acceptance_rate=chain_state.target_acceptance_rate,
            step=chain_state.step,
            extra_stats={**chain_state.extra_stats, "rho": rho_next},
        )
        return next_state, next_chain_state


class TPCNStep(PCNStep):
    """t-preconditioned Crank-Nicolson step.

    This uses a Student-t distribution for the proposal. See
    https://arxiv.org/abs/2407.07781 for details.

    Parameters
    ----------
    dims : int
        Number of dimensions of the target distribution.
    xp : Any
        The array library to use (e.g., numpy, jax.numpy).
    rng_backend : orng.functional.FunctionalBackend
        The random number generator backend to use.
    rho : float, optional
        pCN step size parameter, must be in the range (0, 1). Default is 0.5.
        See https://arxiv.org/abs/2407.07781 for details.
    adaptive : bool, optional
        Whether to adapt the rho parameter during sampling. Default is True.
    """

    def init_state(self, x: Array) -> StepState:
        from .utils import fit_student_t_em

        mu, cov, nu = fit_student_t_em(x)
        if self.dims == 1:
            inv_cov = self.xp.atleast_2d(1.0 / cov)
            chol_cov = self.xp.atleast_2d(self.xp.sqrt(cov))
        else:
            inv_cov = self.xp.linalg.inv(cov)
            chol_cov = self.xp.linalg.cholesky(cov)
        rho = self.xp.asarray(self.rho, dtype=x.dtype)
        return StepState(
            mu=mu,
            cov=cov,
            inv_cov=inv_cov,
            chol_cov=chol_cov,
            rho=rho,
            nu=nu,
        )

    def propose(
        self,
        state: StepState,
        rng_state: Any,
        x: Array,
    ) -> tuple[Any, Array, Array]:
        n_samples = x.shape[0]
        dtype = x.dtype
        diff = x - state.mu
        xx = self.xp.einsum("ni,ij,nj->n", diff, state.inv_cov, diff)
        k = 0.5 * (self.dims + state.nu)
        theta = 2 / (state.nu + xx)

        gamma_draw, rng_state = self.rng_backend.gamma(
            rng_state,
            shape=k,
            scale=theta,
            size=None,
            dtype=dtype,
        )
        z_inv = 1 / gamma_draw

        z, rng_state = self.rng_backend.normal(
            rng_state,
            loc=0.0,
            scale=1.0,
            size=(n_samples, self.dims),
            dtype=dtype,
        )
        scaled_noise = self.xp.sqrt(z_inv)[:, None] * (state.chol_cov @ z.T).T
        rho = state.rho
        x_prime = (
            state.mu
            + self.xp.sqrt(self.xp.asarray(1.0, dtype=dtype) - rho**2) * diff
            + rho * scaled_noise
        )

        diff_prime = x_prime - state.mu
        xx_prime = self.xp.einsum(
            "ni,ij,nj->n", diff_prime, state.inv_cov, diff_prime
        )

        log_a_num = (-0.5 * (state.nu + self.dims)) * self.xp.log1p(
            xx / state.nu
        )
        log_a_denom = (-0.5 * (state.nu + self.dims)) * self.xp.log1p(
            xx_prime / state.nu
        )
        log_alpha = log_a_num - log_a_denom
        return rng_state, x_prime, log_alpha


def step_factory(
    step_name: str,
    dims: int,
    xp: Any,
    rng_backend: Any,
    **kwargs: Any,
) -> Step:
    """Factory function to create a Step instance based on the step name.

    Parameters
    ----------
    step_name : {"pCN", "tPCN"}
        The name of the step type (e.g., "pcn", "tpcn").
    dims : int
        The dimensionality of the samples.
    xp : Any
        The array library to use (e.g., numpy, jax.numpy).
    rng_backend : Any
        The random number generator backend to use (e.g., numpy.random, jax.random).
    **kwargs : Any
        Additional keyword arguments to pass to the step constructor.
    """
    if step_name.lower() == "pcn":
        return PCNStep(dims=dims, xp=xp, rng_backend=rng_backend, **kwargs)
    if step_name.lower() == "tpcn":
        return TPCNStep(
            dims=dims,
            xp=xp,
            rng_backend=rng_backend,
            **kwargs,
        )
    raise ValueError(f"Unknown step type: {step_name}")
