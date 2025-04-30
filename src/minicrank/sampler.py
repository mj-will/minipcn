import numpy as np
from tqdm import trange

from .utils import ChainState, ChainStateHistory


class Sampler:
    def __init__(
        self, log_prob_fn, step_fn, rng, dims, target_acceptance_rate=0.234
    ):
        self.log_prob_fn = log_prob_fn
        self.step_fn = step_fn
        self.rng = rng
        self.dims = dims
        self.target_acceptance_rate = target_acceptance_rate

    def sample(self, x_init: np.ndarray, n_steps: int):
        x = x_init
        self.step_fn.initialise(x)
        log_prob_x = self.log_prob_fn(x)  # Shape: (N,)
        chain = np.empty((n_steps + 1, x.shape[0], x.shape[1]), dtype=x.dtype)
        chain[0] = x
        states = []
        for i in trange(n_steps):
            x_new, log_jac = self.step_fn(x)
            log_prob_x_new = self.log_prob_fn(x_new)
            log_alpha = log_prob_x_new + log_jac - log_prob_x
            alpha = np.exp(np.minimum(0, log_alpha))  # Shape: (N,)

            accept = self.rng.uniform(size=len(x_new)) < alpha
            x = np.where(accept[:, None], x_new, x)  # Shape: (N, D)
            log_prob_x = np.where(accept, log_prob_x_new, log_prob_x)
            chain[i + 1] = x

            state = ChainState(
                it=i,
                acceptance_rate=float(np.mean(accept)),
                target_acceptance_rate=self.target_acceptance_rate,
            )
            self.step_fn.update(state)
            state = self.step_fn.update_state(state)
            states.append(state)

        state_history = ChainStateHistory.from_chain_states(states)
        return chain, state_history
