from __future__ import annotations

from typing import Any, Callable
from warnings import warn

import numpy as np
from orng import RandomGenerator
from orng.functional import (
    create_functional_backend,
    create_functional_backend_from_xp,
)
from tqdm import trange

from ._typing import Array
from .step import Step
from .utils import ChainState, ChainStateHistory, _is_jax_tracer, _to_scalar


class Sampler:
    """Class for running the MiniPCN sampler.

    Parameters
    ----------
    log_prob_fn : Callable
        Function to compute the log probability of the target distribution.
        It should take a single argument (the samples) and return the log
        probability.
    step_fn : Step | str
        Step object that defines the proposal distribution and the
        transformation to the target distribution. If a string is provided,
        it should be the name of a known step type (e.g., "pCN" or "tpCN").
    rng : np.random.Generator | ArrayRNG
        Random number generator for reproducibility.
    dims : int
        Number of dimensions of the target distribution.
    target_acceptance_rate : float, optional
        Target acceptance rate for the sampler. Default is 0.234.
    xp : Any, optional
        Array namespace to use (e.g., numpy, jax.numpy, torch). Default is numpy.
    **kwargs
        Additional keyword arguments to pass to the step function if `step_fn`
        is provided as a string.
    """

    def __init__(
        self,
        log_prob_fn: Callable,
        step_fn: str,
        dims: int,
        target_acceptance_rate: float = 0.234,
        xp: Any = np,
        rng: Any | None = None,
        **kwargs,
    ) -> None:
        self.log_prob_fn = log_prob_fn

        if not isinstance(step_fn, str):
            raise TypeError(
                "step_fn must be the name of a built-in step. "
                "Pass 'pcn' or 'tpcn'."
            )

        self._step_name = step_fn
        self._step_kwargs = kwargs

        self.dims = dims
        self.target_acceptance_rate = target_acceptance_rate
        self.xp = xp

        if rng is not None:
            warn(
                "Passing rng to the Sampler constructor is deprecated and "
                "will be removed in a future version. "
                "Please pass the rng directly to the `sample` method instead.",
                UserWarning,
            )
        self.rng = rng

    def _resolve_rng(
        self, rng: Any | None, seed: int | None
    ) -> tuple[Any, Any]:
        if self.rng is not None:
            if rng is not None:
                warn(
                    "Both the Sampler constructor and the sample method were given "
                    "an rng. The rng passed to the sample method will be used.",
                    UserWarning,
                )
            else:
                rng = self.rng

        if rng is None:
            rng_backend = create_functional_backend("numpy", pure=False)
            rng_state = rng_backend.init_state(seed=seed, generator=None)
        elif isinstance(rng, np.random.Generator):
            rng_backend = create_functional_backend("numpy", pure=False)
            rng_state = rng_backend.init_state(seed=seed, generator=rng)
        elif isinstance(rng, RandomGenerator):
            rng_backend, rng_state = rng.to_functional()
        else:
            raise TypeError(
                "rng must be a numpy.random.Generator or an orng.RandomGenerator. "
                "For explicit functional RNG state, use `sample_functional()`."
            )
        return rng_backend, rng_state

    def sample(
        self,
        x_init: Array,
        n_steps: int,
        *,
        rng: RandomGenerator | np.random.Generator = None,
        seed: int | None = None,
        verbose: bool = True,
        return_last_only: bool = False,
    ) -> tuple[Array, ChainStateHistory]:
        """Run the minipcn sampler.

        Parameters
        ----------
        x_init:
            Initial sample or batch of samples. A one-dimensional input is
            promoted to a batch of size one.
        n_steps:
            Number of MCMC steps to run.
        rng:
            Stateful RNG object. Supported inputs are
            ``numpy.random.Generator`` and ``orng.RandomGenerator``.
        verbose:
            If ``True``, display a progress bar with per-step diagnostics.
        return_last_only:
            If ``True``, return only the final sample batch. Otherwise return
            the full chain history stacked along axis 0.

        Returns
        -------
        chain:
            Either the final sample batch or the full chain, depending on
            ``return_last_only``.
        history:
            Per-step diagnostics collected during sampling.
        """
        rng_backend, rng_state = self._resolve_rng(rng, seed=seed)

        chain, history, next_rng_state = self._sample_impl(
            x_init=x_init,
            n_steps=n_steps,
            rng_backend=rng_backend,
            rng_state=rng_state,
            verbose=verbose,
            return_last_only=return_last_only,
        )

        if hasattr(rng, "_impl") and not _is_jax_tracer(next_rng_state):
            rng._impl._state = next_rng_state

        return chain, history

    def sample_functional(
        self,
        x_init: Array,
        n_steps: int,
        *,
        rng_state: Any,
        verbose: bool = True,
        return_last_only: bool = False,
    ) -> tuple[Array, ChainStateHistory, Any]:
        """Run the minipcn sampler with an explicit functional RNG state.

        This method is intended for functional workflows, including JAX
        compilation, where RNG state is threaded explicitly through the
        sampling loop.

        Parameters
        ----------
        x_init:
            Initial sample or batch of samples. A one-dimensional input is
            promoted to a batch of size one.
        n_steps:
            Number of MCMC steps to run.
        rng_state:
            Backend-native functional RNG state. The matching functional
            backend is inferred from ``self.xp``.
        verbose:
            If ``True``, display a progress bar with per-step diagnostics.
        return_last_only:
            If ``True``, return only the final sample batch. Otherwise return
            the full chain history stacked along axis 0.

        Returns
        -------
        chain:
            Either the final sample batch or the full chain, depending on
            ``return_last_only``.
        history:
            Per-step diagnostics collected during sampling.
        next_rng_state:
            Updated functional RNG state after the final step.
        """
        rng_backend = create_functional_backend_from_xp(self.xp)
        return self._sample_impl(
            x_init=x_init,
            n_steps=n_steps,
            rng_backend=rng_backend,
            rng_state=rng_state,
            verbose=verbose,
            return_last_only=return_last_only,
        )

    def _sample_impl(
        self,
        *,
        x_init: Array,
        n_steps: int,
        rng_backend: Any,
        rng_state: Any,
        verbose: bool,
        return_last_only: bool,
    ) -> tuple[Array, ChainStateHistory, Any]:
        x = self.xp.atleast_2d(x_init)
        step_fn = self._get_step(rng_backend)
        step_state = step_fn.init_state(x)
        log_prob_x = self.log_prob_fn(x)

        chain_states: list[Array] | None
        if return_last_only:
            chain_states = None
        else:
            chain_states = [x]

        history_states: list[ChainState] = []

        iterator = range(n_steps)
        if verbose:
            iterator = trange(n_steps, desc="Sampling", unit="step")

        for i in iterator:
            rng_state, x_new, log_alpha_step = step_fn.propose(
                step_state,
                rng_state,
                x,
            )
            log_prob_x_new = self.log_prob_fn(x_new)
            log_alpha = log_prob_x_new - log_prob_x + log_alpha_step
            alpha = self.xp.exp(
                self.xp.minimum(
                    self.xp.asarray(0.0, dtype=log_alpha.dtype),
                    log_alpha,
                )
            )

            uniform, rng_state = rng_backend.uniform(
                rng_state,
                low=0.0,
                high=1.0,
                size=(x_new.shape[0],),
                dtype=x_new.dtype,
            )
            accept = uniform < alpha
            x = self.xp.where(accept[:, None], x_new, x)
            log_prob_x = self.xp.where(accept, log_prob_x_new, log_prob_x)

            if chain_states is not None:
                chain_states.append(x)

            acceptance_rate = self.xp.sum(accept) / accept.shape[0]
            chain_state = ChainState(
                it=i,
                acceptance_rate=acceptance_rate,
                target_acceptance_rate=self.target_acceptance_rate,
                step=step_fn.step_name,
            )
            step_state, chain_state = step_fn.adapt(
                step_state,
                chain_state,
                samples=x,
            )
            history_states.append(chain_state)

            if verbose:
                iterator.set_postfix(
                    {
                        "acceptance_rate": _to_scalar(
                            chain_state.acceptance_rate
                        ),
                        **{
                            key: _to_scalar(value)
                            for key, value in chain_state.extra_stats.items()
                        },
                    }
                )

        if return_last_only:
            chain = x
        else:
            chain = self.xp.stack(chain_states, axis=0)

        if history_states:
            history = ChainStateHistory.from_chain_states(
                history_states, xp=self.xp
            )
        else:
            history = ChainStateHistory(
                it=[],
                acceptance_rate=[],
                target_acceptance_rate=[],
                step=step_fn.step_name,
                extra_stats={},
            )
        return chain, history, rng_state

    def _get_step(self, rng_backend: Any) -> Step:
        from .step import step_factory

        return step_factory(
            self._step_name,
            self.dims,
            self.xp,
            rng_backend=rng_backend,
            **self._step_kwargs,
        )
