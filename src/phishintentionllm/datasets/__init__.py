from .base import BaseDatasetLoader
from .phishiris import PhishIrisLoader
from .putra import PutraLoader
from .registry import DatasetRegistry

__all__ = ["BaseDatasetLoader", "PutraLoader", "PhishIrisLoader", "DatasetRegistry"]
