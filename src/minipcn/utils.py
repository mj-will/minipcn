from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
from array_api_compat import array_namespace, is_jax_namespace

from ._typing import Array
from .student_t import fit_student_t_em as fit_student_t_em

try:
    from ._jax import _register_dataclass as register_dataclass
except ImportError:

    def register_dataclass(*args, **kwargs):
        def decorator(cls):
            return cls

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return decorator


@register_dataclass
@dataclass
class ChainState:
    """State of the chain at a given iteration.

    Attributes
    ----------
    it : Any
        Current iteration number or backend scalar.
    acceptance_rate : Any
        Acceptance rate of the current iteration.
    target_acceptance_rate : Any
        Target acceptance rate for the chain.
    step : str
        Name of the step function used in this iteration.
    extra_stats : Dict[str, Any]
        Additional statistics collected during the iteration.
    """

    it: Any
    acceptance_rate: Any
    target_acceptance_rate: Any
    step: str = field(default="", metadata={"static": True})
    extra_stats: Dict[str, Any] = field(default_factory=dict)


@register_dataclass
@dataclass
class ChainStateHistory:
    it: Any
    acceptance_rate: Any
    target_acceptance_rate: Any
    step: str = field(default="", metadata={"static": True})
    extra_stats: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, index: Union[int, slice]) -> "ChainStateHistory":
        # Support slicing or single index
        if isinstance(index, int):
            return ChainStateHistory(
                it=_history_single_item(self.it, index),
                acceptance_rate=_history_single_item(
                    self.acceptance_rate, index
                ),
                target_acceptance_rate=_history_single_item(
                    self.target_acceptance_rate, index
                ),
                step=self.step,
                extra_stats={
                    k: _history_single_item(v, index)
                    for k, v in self.extra_stats.items()
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
        cls, states: List[ChainState], xp: Any | None = None
    ) -> "ChainStateHistory":
        if not states:
            return cls(
                it=[],
                acceptance_rate=[],
                target_acceptance_rate=[],
                extra_stats={},
            )

        step = states[0].step
        extra_stats = {
            key: [s.extra_stats[key] for s in states]
            for key in states[0].extra_stats.keys()
        }
        history_data = {
            "it": [s.it for s in states],
            "acceptance_rate": [s.acceptance_rate for s in states],
            "target_acceptance_rate": [
                s.target_acceptance_rate for s in states
            ],
            "extra_stats": extra_stats,
        }
        if xp is not None and is_jax_namespace(xp):
            from ._jax import _contains_tracer, _stack_history_values

            if _contains_tracer(history_data):
                return cls(
                    it=_stack_history_values(history_data["it"], xp),
                    acceptance_rate=_stack_history_values(
                        history_data["acceptance_rate"], xp
                    ),
                    target_acceptance_rate=_stack_history_values(
                        history_data["target_acceptance_rate"], xp
                    ),
                    step=step,
                    extra_stats={
                        key: _stack_history_values(values, xp)
                        for key, values in extra_stats.items()
                    },
                )

        return cls(
            it=[
                int(to_numpy_array(value).reshape(-1)[0])
                for value in history_data["it"]
            ],
            acceptance_rate=[
                float(to_numpy_array(value).reshape(-1)[0])
                for value in history_data["acceptance_rate"]
            ],
            target_acceptance_rate=[
                float(to_numpy_array(value).reshape(-1)[0])
                for value in history_data["target_acceptance_rate"]
            ],
            step=step,
            extra_stats={
                key: [
                    float(to_numpy_array(value).reshape(-1)[0])
                    for value in values
                ]
                for key, values in extra_stats.items()
            },
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


def _history_single_item(value: Any, index: int) -> Any:
    if isinstance(value, list):
        return [value[index]]
    return value[index : index + 1]


def _to_scalar(value: Any) -> float:
    try:
        return float(to_numpy_array(value).reshape(-1)[0])
    except Exception:
        from ._jax import _is_tracer

        if _is_tracer(value):
            return float("nan")
        raise


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
