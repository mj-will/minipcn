from minipcn import Sampler
from minipcn.step import TPCNStep


def test_sampling_with_tcpn(rng, log_target_fn):
    dims = 4
    x_init = rng.normal(size=(100, dims))  # Initial samples

    sampler = Sampler(
        log_prob_fn=log_target_fn,
        dims=dims,
        step_fn=TPCNStep(dims=dims, rng=rng, rho=0.5),
        rng=rng,
        target_acceptance_rate=0.234,
    )

    chain, history = sampler.sample(x_init, n_steps=100)
    assert chain.shape == (101, 100, 4)
    assert history.it[-1] == 99
