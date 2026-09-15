from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable
from warnings import warn

import numpy as np
from array_api_compat import array_namespace, device, is_jax_namespace
from orng import RandomGenerator, infer_backend_name_from_xp
from orng.functional import (
    create_functional_backend,
    create_functional_backend_from_xp,
)
from tqdm import trange

from ._typing import Array
from .step import Step, StepState
from .utils import ChainState, ChainStateHistory, _to_scalar


class Sampler:
    """Class for running the MiniPCN sampler.

    Parameters
    ----------
    log_prob_fn : Callable
        Function to compute the log probability of the target distribution.
        It should take a single argument (the samples) and return the log
        probability.
    step_fn : str
        Name of the step type to use (e.g., "pCN" or "tpCN").
    rng : np.random.Generator | RandomGenerator
        Random number generator for reproducibility.
    dims : int
        Number of dimensions of the target distribution.
    target_acceptance_rate : float, optional
        Target acceptance rate for the sampler. Default is 0.234.
    xp : Any, optional
        Array namespace used for sampling (e.g., numpy, jax.numpy, torch).
        Samples and RNG must use this backend. Default is numpy.
    **kwargs
        Additional keyword arguments to pass to the step function.
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
        self,
        rng: Any | None,
        seed: int | None,
        *,
        rng_device: Any | None = None,
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

        backend_name = infer_backend_name_from_xp(self.xp)
        if rng is None:
            rng_name = backend_name
        elif isinstance(rng, np.random.Generator):
            rng_name = "numpy"
        elif isinstance(rng, RandomGenerator):
            rng_name = rng.backend.lower()
            if rng_name == "pytorch":
                rng_name = "torch"
        else:
            # The type error below handles unsupported RNGs.
            rng_name = None
        if rng_name is not None and rng_name != backend_name:
            raise ValueError(
                f"RNG backend '{rng_name}' does not match the sampler "
                f"backend '{backend_name}'. Supply an RNG with the "
                "same backend."
            )

        if rng is None:
            rng_backend = create_functional_backend(
                backend_name, device=rng_device, pure=backend_name == "jax"
            )
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

    def _validate_array_backend(self, x: Array) -> None:
        array_backend = infer_backend_name_from_xp(array_namespace(x))
        sampler_backend = infer_backend_name_from_xp(self.xp)
        if array_backend != sampler_backend:
            raise ValueError(
                f"Input array backend '{array_backend}' does not match the "
                f"sampler backend '{sampler_backend}'. Configure xp to match "
                "the samples."
            )

    def init_step_state(self, x_init: Array) -> StepState:
        """Fit proposal parameters without consuming random state.

        Parameters
        ----------
        x_init : array, shape (n_samples, dims)
            Samples used to fit the Gaussian or Student-t proposal. As in
            sampling, a one-dimensional input is promoted to one sample.

        Returns
        -------
        step_state : StepState
            Fitted proposal parameters with an iteration count of zero. Pass
            this state to ``sample`` or ``sample_functional`` to skip fitting.
            It contains neither particles nor cached target evaluations.

        Notes
        -----
        Supports JAX tracing with the sampler configuration held static. The
        temporary step and RNG backend use the namespace of ``x_init``,
        independently of ``self.xp``. Fitting does not draw random numbers
        or initialize a random state. Sampling must subsequently use a
        compatible array backend.
        """
        xp = array_namespace(x_init)
        x = xp.atleast_2d(x_init)
        rng_backend = create_functional_backend_from_xp(xp)
        step = self._get_step(rng_backend, xp=xp)
        return step.init_state(x)

    def sample(
        self,
        x_init: Array,
        n_steps: int,
        *,
        rng: RandomGenerator | np.random.Generator | None = None,
        seed: int | None = None,
        verbose: bool = True,
        return_last_only: bool = False,
        step_state: StepState | None = None,
        return_step_state: bool = False,
    ) -> (
        tuple[Array, ChainStateHistory]
        | tuple[Array, ChainStateHistory, StepState]
    ):
        """Run the minipcn sampler.

        Parameters
        ----------
        x_init:
            Initial sample or batch of samples. A one-dimensional input is
            promoted to a batch of size one. Its backend must match ``self.xp``.
        n_steps:
            Number of MCMC steps to run.
        rng:
            Stateful RNG object. Supported inputs are
            ``numpy.random.Generator`` and ``orng.RandomGenerator``. Its backend
            must match ``self.xp``; mismatches raise ValueError before fitting
            or target evaluation. When no RNG is supplied here or at
            construction, one is created for ``self.xp`` on the input device.
        verbose:
            If ``True``, display a progress bar with per-step diagnostics.
        return_last_only:
            If ``True``, return only the final sample batch. Otherwise return
            the full chain history stacked along axis 0.
        step_state : StepState, optional
            Previously fitted state from this sampler's step type, dimensions,
            array backend and dtype. If supplied, reuse its parameters without
            fitting. The target is always reevaluated at ``x_init``, which may
            have changed since the preceding call. Adaptation continues from
            the state's iteration count; use ``adaptive=False`` to freeze the
            proposal for interleaved moves.
        return_step_state : bool, optional
            Append the final StepState to the result tuple. Default False
            preserves the existing return signature. Under JAX JIT, this flag
            must be static, as must ``n_steps`` and ``return_last_only``.

        Returns
        -------
        chain:
            Either the final sample batch or the full chain, depending on
            ``return_last_only``.
        history:
            Per-step diagnostics collected during sampling. Iteration numbers
            continue from the supplied state's iteration count.
        final_step_state : StepState, optional
            Final proposal parameters and iteration count, returned only when
            ``return_step_state=True``. The input state is not mutated.
        """
        self._validate_array_backend(x_init)
        xp = self.xp
        rng_backend, rng_state = self._resolve_rng(
            rng,
            seed=seed,
            rng_device=device(x_init),
        )

        chain, history, next_rng_state, final_step_state = self._sample_impl(
            x_init=x_init,
            n_steps=n_steps,
            rng_backend=rng_backend,
            rng_state=rng_state,
            verbose=verbose,
            return_last_only=return_last_only,
            step_state=step_state,
        )

        is_tracer = False
        if is_jax_namespace(xp):
            from ._jax import _is_tracer

            is_tracer = _is_tracer(next_rng_state)
        if hasattr(rng, "_impl") and not is_tracer:
            rng._impl._state = next_rng_state

        if return_step_state:
            return chain, history, final_step_state
        return chain, history

    def sample_functional(
        self,
        x_init: Array,
        n_steps: int,
        *,
        rng_state: Any,
        verbose: bool = True,
        return_last_only: bool = False,
        step_state: StepState | None = None,
        return_step_state: bool = False,
    ) -> (
        tuple[Array, ChainStateHistory, Any]
        | tuple[Array, ChainStateHistory, Any, StepState]
    ):
        """Run the minipcn sampler with an explicit functional RNG state.

        This method is intended for functional workflows, including JAX
        compilation, where RNG state is threaded explicitly through the
        sampling loop.

        Parameters
        ----------
        x_init:
            Initial sample or batch of samples. A one-dimensional input is
            promoted to a batch of size one. Its backend must match ``self.xp``.
        n_steps:
            Number of MCMC steps to run.
        rng_state:
            Backend-native functional RNG state. The matching functional
            backend is determined by ``self.xp``. The supplied state must
            belong to that backend.
        verbose:
            If ``True``, display a progress bar with per-step diagnostics.
        return_last_only:
            If ``True``, return only the final sample batch. Otherwise return
            the full chain history stacked along axis 0.
        step_state : StepState, optional
            Previously fitted state from this sampler's step type, dimensions,
            array backend and dtype. If supplied, reuse its parameters without
            fitting. The target is always reevaluated at ``x_init``, which may
            have changed since the preceding call. Adaptation continues from
            the state's iteration count; use ``adaptive=False`` to freeze the
            proposal for interleaved moves.
        return_step_state : bool, optional
            Append the final StepState to the result tuple. Default False
            preserves the existing return signature. Under JAX JIT, this flag
            must be static, as must ``n_steps`` and ``return_last_only``.

        Returns
        -------
        chain:
            Either the final sample batch or the full chain, depending on
            ``return_last_only``.
        history:
            Per-step diagnostics collected during sampling.
        next_rng_state:
            Updated functional RNG state after the final step.
        final_step_state : StepState, optional
            Final proposal parameters and iteration count, appended after
            ``next_rng_state`` only when ``return_step_state=True``. This is a
            JAX pytree; the input state is not mutated.
        """
        self._validate_array_backend(x_init)
        rng_backend = create_functional_backend_from_xp(
            self.xp, device=device(x_init)
        )
        chain, history, next_rng_state, final_step_state = self._sample_impl(
            x_init=x_init,
            n_steps=n_steps,
            rng_backend=rng_backend,
            rng_state=rng_state,
            verbose=verbose,
            return_last_only=return_last_only,
            step_state=step_state,
        )
        if return_step_state:
            return chain, history, next_rng_state, final_step_state
        return chain, history, next_rng_state

    def _sample_impl(
        self,
        *,
        x_init: Array,
        n_steps: int,
        rng_backend: Any,
        rng_state: Any,
        verbose: bool,
        return_last_only: bool,
        step_state: StepState | None = None,
    ) -> tuple[Array, ChainStateHistory, Any, StepState]:
        xp = self.xp
        x = xp.atleast_2d(x_init)
        step_fn = self._get_step(rng_backend)
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative.")
        if step_state is None:
            step_state = step_fn.init_state(x)
        elif not isinstance(step_state, StepState):
            raise TypeError("step_state must be a StepState or None.")
        initial_iteration = step_state.iteration
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
            alpha = xp.exp(
                xp.minimum(
                    xp.asarray(0.0, dtype=log_alpha.dtype),
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
            x = xp.where(accept[:, None], x_new, x)
            log_prob_x = xp.where(accept, log_prob_x_new, log_prob_x)

            if chain_states is not None:
                chain_states.append(x)

            acceptance_rate = xp.sum(accept) / accept.shape[0]
            chain_state = ChainState(
                it=initial_iteration + i,
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
            chain = xp.stack(chain_states, axis=0)

        if history_states:
            history = ChainStateHistory.from_chain_states(
                history_states, xp=xp
            )
        else:
            history = ChainStateHistory(
                it=[],
                acceptance_rate=[],
                target_acceptance_rate=[],
                step=step_fn.step_name,
                extra_stats={},
            )
        final_step_state = replace(
            step_state, iteration=initial_iteration + n_steps
        )
        return chain, history, rng_state, final_step_state

    def _get_step(self, rng_backend: Any, *, xp: Any | None = None) -> Step:
        from .step import step_factory

        return step_factory(
            self._step_name,
            self.dims,
            self.xp if xp is None else xp,
            rng_backend=rng_backend,
            **self._step_kwargs,
        )
