# minipcn

[![DOI](https://zenodo.org/badge/975531339.svg)](https://doi.org/10.5281/zenodo.15657997)

A minimal implementation of preconditioned Crank-Nicolson MCMC sampling.

## Installation

`minipcn` can be installed using from PyPI using `pip`:

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
    step_fn="pcn",    # Or tpcn
    rng=rng,
)

# Generate initial samples
x0 = rng.randn(size=(100, dims))

# Run the sampler
chain, history = sampler.run(x0, n_steps=500)
```

For a complete example, see the `examples` directory.

## Support for array-api

`minipcn` also supports different array API backends via `array-api-compat`
and `orng` for random number generation. These can be installed by running

```
pip install minicpn[array-api]
```

Usage is then similar to when using numpy, except one must use the RNG from
`orng` and specify the backend via `xp`:

```python
from minipcn import Sampler
from orng import ArrayRNG
import torch

log_prob_fn = ...    # Log-probability function - must be vectorized
dims = ...    # The number of dimensions
rng = ArrayRNG(backend="torch", seed=42)

sampler = Sampler(
    log_prob_fn=log_prob_fn,
    dims=dims,
    step_fn="pcn",    # Or tpcn
    rng=rng,
    xp=torch,
)

# Generate initial samples
x0 = rng.randn(size=(100, dims))

# Run the sampler
chain, history = sampler.run(x0, n_steps=500)

```
**Note:** this still uses `numpy` when initializing the `tpcn` kernel.


## Citing minipcn

If you use `minipcn` in your work, please cite our [DOI](https://doi.org/10.5281/zenodo.15657997)

If using the `tpcn` kernel, please also cite [Grumitt et al](https://arxiv.org/abs/2407.07781)
