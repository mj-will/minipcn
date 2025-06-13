"""Example of using the minipcn package for sampling from a target distribution.

This example demonstrates how to set up a sampler, define a target distribution,
and visualize the results using corner plots and acceptance rate history.
"""

import corner
import numpy as np

from minipcn import Sampler

# Set up the random number generator and dimensions
rng = np.random.default_rng(seed=42)

# Define the target distribution parameters
dims = 4


# Define the log target function
# NOTE: the log target function must be vectorized and accept inputs of shape
# (n_samples, dims)
def log_target_fn(x):
    logl = -np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
        + (1.0 - x[..., :-1]) ** 2.0,
        axis=-1,
    )
    # In [-5, 5]^dims bounds
    logl = np.where(
        np.all(np.abs(x) <= 5, axis=-1),
        logl,
        -np.inf,
    )
    return logl


# Choose the step function
step_fn = "tpCN"  # Choose between "tpCN" or "pCN"
# Generate initial samples
x_init = rng.uniform(-5, 5, size=(100, dims))  # Initial samples

sampler = Sampler(
    log_prob_fn=log_target_fn,
    dims=dims,
    step_fn=step_fn,
    rng=rng,
    target_acceptance_rate=0.234,  # Target acceptance rate for the sampler
)

# Sample from the target distribution using the sampler
# The chain will be shape (n_iterations, n_samples, dims)
# The history will be a ChainStateHistory object containing the acceptance
# rate and other statistics
chain, history = sampler.sample(x_init, n_steps=1000)

# Discard burn-in samples
burn_in = 100  # Number of burn-in steps
chain = chain[burn_in:]  # Discard burn-in samples
history = history[burn_in:]  # Discard burn-in history

# Visualize the results
# Plot the acceptance rate history
fig = history.plot_acceptance_rate()
fig.savefig(f"acceptance_rate_{step_fn}.png")

# Plot the rho history -  this is a parameter for the pCN step function
fig = history.plot_extra_stat("rho")
fig.savefig(f"rho_{step_fn}.png")

# Plot the corner plot step of the chain
fig = corner.corner(chain[-1])
fig.savefig(f"corner_plot_{step_fn}.png")
