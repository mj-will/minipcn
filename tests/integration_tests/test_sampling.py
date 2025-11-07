from minipcn import Sampler


def test_sampling(rng, log_target_fn, step_fn, dims, xp):
    x_init = rng.normal(size=(100, dims))  # Initial samples

    sampler = Sampler(
        log_prob_fn=log_target_fn,
        dims=dims,
        step_fn=step_fn,
        rng=rng,
        target_acceptance_rate=0.234,
        xp=xp,
    )

    chain, history = sampler.sample(x_init, n_steps=100)
    assert chain.shape == (101, 100, dims)
    assert history.it[-1] == 99
