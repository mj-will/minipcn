# minipcn

A minimalistic implementation of preconditioned Crank-Nicolson MCMC sampling.

## Installation

Currently, the `minipcn` is only available to install from source. Clone the
repo and then run:

```bash
pip install .
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
x0 = rng.randn(size=(100, dims)

# Run the sampler
chain, history = sampler.run(x0, n_steps=500)
```

For a complete example, see the `examples` directory.
