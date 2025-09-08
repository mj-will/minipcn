"""minipcn

A minimal implementation of preconditioned Crank-Nicolson (pCN) and
t-preconditioned Crank-Nicolson (tpCN) MCMC samplers for Bayesian inference.
"""

from importlib.metadata import PackageNotFoundError, version

from .sampler import Sampler

try:
    __version__ = version("minipcn")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["Sampler"]
