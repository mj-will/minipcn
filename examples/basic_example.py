import corner
import numpy as np
from scipy import stats

from minipcn import Sampler
from minipcn.step import TPCNStep

rng = np.random.default_rng(seed=42)

dims = 4
dist = stats.multivariate_normal(
    mean=2.0 * np.ones(dims), cov=0.1 * np.eye(dims)
)


def log_target_fn(x):
    return dist.logpdf(x)


x_init = rng.normal(size=(1000, dims))  # Initial samples

sampler = Sampler(
    log_prob_fn=log_target_fn,
    dims=dims,
    step_fn=TPCNStep(dims=dims, rng=rng, rho=0.5),
    rng=rng,
    target_acceptance_rate=0.234,
)

chain, history = sampler.sample(x_init, n_steps=1000)

fig = history.plot_acceptance_rate()
fig.savefig("acceptance_rate.png")

fig = history.plot_extra_stat("rho")
fig.savefig("rho.png")

target_samples = dist.rvs(size=len(x_init))

fig = corner.corner(target_samples, color="k")
fig = corner.corner(chain[-1], color="red", fig=fig)
fig.savefig("corner_plot.png")
