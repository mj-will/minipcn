from importlib.metadata import PackageNotFoundError, version

from .sampler import Sampler

try:
    __version__ = version("minipcn")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["Sampler"]
