from importlib.metadata import PackageNotFoundError, version

from .sampler import Sampler

try:
    __version__ = version("minicrank")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["Sampler"]
