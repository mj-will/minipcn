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

## Citing minipcn

If you use `minipcn` in your work, please cite our [DOI](https://doi.org/10.5281/zenodo.15657997)

If using the `tpcn` kernel, please also cite [Grumitt et al](https://arxiv.org/abs/2407.07781)
