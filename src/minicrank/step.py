import numpy as np

from .utils import ChainState


class Step:
    def __init__(self, dims: int, rng: np.random.Generator):
        self.dims = dims
        self.rng = rng

    def initialise(self, x: np.ndarray):
        pass

    def update(self, state: ChainState):
        pass

    def update_state(self, state: ChainState) -> ChainState:
        state.step = self.__class__.__name__
        return state

    def step(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


class TPCNStep(Step):
    def __init__(self, dims, rng, rho: float = 0.5):
        super().__init__(dims, rng)

        if not (0 < rho < 1):
            raise ValueError("rho must be in the range (0, 1).")
        self.rho = rho

    def initialise(self, x):
        from .utils import fit_student_t_em

        self.mu, self.cov, self.nu = fit_student_t_em(x)
        self.inv_cov = np.linalg.inv(self.cov)
        self.chol_cov = np.linalg.cholesky(self.cov)

    def update(self, state):
        self.rho = np.abs(
            np.minimum(
                self.rho
                + 1
                / (state.it + 1) ** 0.75
                * (state.acceptance_rate - state.target_acceptance_rate),
                np.minimum(2.38 / self.dims**0.5, 0.99),
            )
        )

    def update_state(self, state):
        state = super().update_state(state)
        state.extra_stats["rho"] = self.rho
        return state

    def step(self, x):
        n_samples = x.shape[0]
        diff = x - self.mu  # Shape: (N, D)

        # Mahalanobis distances
        xx = np.einsum("ni,ij,nj->n", diff, self.inv_cov, diff)  # Shape: (N,)
        k = 0.5 * (self.dims + self.nu)
        theta = 2 / (self.nu + xx)
        z_inv = 1 / self.rng.gamma(shape=k, scale=theta)  # Shape: (N,)

        # Propose new samples
        z = self.rng.normal(size=(n_samples, self.dims))
        scaled_noise = (
            (np.sqrt(z_inv)[:, None]) * (self.chol_cov @ z.T).T
        )  # Shape: (N, D)
        x_prime = (
            self.mu + np.sqrt(1 - self.rho**2) * diff + self.rho * scaled_noise
        )  # Shape: (N, D)

        diff_prime = x_prime - self.mu
        xx_prime = np.einsum(
            "ni,ij,nj->n", diff_prime, self.inv_cov, diff_prime
        )

        log_a_num = (-0.5 * (self.nu + self.dims)) * np.log1p(xx / self.nu)
        log_a_denom = (-0.5 * (self.nu + self.dims)) * np.log1p(
            xx_prime / self.nu
        )
        log_alpha = log_a_num - log_a_denom
        return x_prime, log_alpha
