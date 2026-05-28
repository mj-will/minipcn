# minipcn

[![DOI](https://zenodo.org/badge/975531339.svg)](https://doi.org/10.5281/zenodo.15657997)

A minimal implementation of preconditioned Crank-Nicolson MCMC sampling.

## Installation

`minipcn` can be installed from PyPI using `pip`:

```bash
pip install minipcn
```

## Usage

The basic usage is:

```python
from minipcn import Sampler
import numpy as np

log_prob_fn = ...    # Log-probability function - must be vectorized
dims = ...    # The number of dimensions
rng = np.random.default_rng(42)

sampler = Sampler(
    log_prob_fn=log_prob_fn,
    dims=dims,
    step_fn="pcn",  # Or "tpcn"
)

x0 = rng.normal(size=(100, dims))

chain, history = sampler.sample(x0, n_steps=500, rng=rng)
```

For a complete example, see the `examples` directory.

## Array API support

`minipcn` also supports different array API backends via `array-api-compat`
and `orng` for random number generation. These can be installed by running

```
pip install minicpn[array-api]
```

Usage is then similar to when using numpy, except one must use the RNG from
`orng` and specify the backend via `xp`:

```python
from minipcn import Sampler
from orng import RandomGenerator
import torch

log_prob_fn = ...    # Log-probability function - must be vectorized
dims = ...    # The number of dimensions
rng = RandomGenerator(backend="torch", seed=42)

sampler = Sampler(
    log_prob_fn=log_prob_fn,
    dims=dims,
    step_fn="pcn",    # Or tpcn
    xp=torch,
)

# Generate initial samples
x0 = rng.randn(size=(100, dims))

# Run the sampler
chain, history = sampler.run(x0, n_steps=500, rng=rng)

```

**Note:** the tpCN step falls back to numpy for fitting the Student-t distribution

## Functional API

`minipcn` also supports explicit functional RNG state via
`Sampler.sample_functional(...)`. This is the path to use for JAX compilation
or any workflow where RNG state must be threaded explicitly.

The functional API does not take an RNG object but a backend and state:


```python
import jax
import jax.numpy as jnp
from minipcn import Sampler
from orng.functional import create_functional_backend

dims = 4
rng_backend = create_functional_backend("jax")
rng_state = rng_backend.init_state(seed=42, generator=None)
x0, rng_state = rng_backend.normal(
    rng_state,
    loc=0.0,
    scale=1.0,
    size=(32, dims),
    dtype=jnp.float32,
)

def log_prob_fn(x):
    return -0.5 * jnp.sum(x**2, axis=-1)

sampler = Sampler(
    log_prob_fn=log_prob_fn,
    dims=dims,
    step_fn="pcn",
    xp=jnp,
)

samples, history, next_rng_state = sampler.sample_functional(
    x0,
    n_steps=8,
    rng_state=rng_state,
    verbose=False,
    return_last_only=True,
)
```

`sample_functional(...)` returns `(chain, history, next_rng_state)`.

To use it under `jax.jit`, thread the state through the compiled function:

```python
@jax.jit
def run(x, state):
    samples, history, next_state = sampler.sample_functional(
        x,
        n_steps=8,
        rng_state=state,
        verbose=False,
        return_last_only=True,
    )
    return samples, history, next_state

samples, history, rng_state = run(x0, rng_state)
```

The backend for `sample_functional(...)` is inferred from `xp`. For example:

- `xp=np` uses the NumPy functional backend
- `xp=jax.numpy` uses the JAX functional backend
- `xp=torch` uses the PyTorch functional backend

Use `sample(...)` for stateful RNG objects and `sample_functional(...)` when
you want explicit RNG state.

## Citing minipcn

If you use `minipcn` in your work, please cite our [DOI](https://doi.org/10.5281/zenodo.15657997)

If using the `tpcn` kernel, please also cite [Grumitt et al](https://arxiv.org/abs/2407.07781)
