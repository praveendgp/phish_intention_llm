"""Unified access to every configured dataset source."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Type

from ..config import Config, load_config
from ..schemas import Sample
from ..utils.logging import get_logger
from .base import BaseDatasetLoader
from .phishiris import PhishIrisLoader
from .putra import PutraLoader

log = get_logger(__name__)

LOADERS: Dict[str, Type[BaseDatasetLoader]] = {
    PutraLoader.source_name: PutraLoader,
    PhishIrisLoader.source_name: PhishIrisLoader,
}


class DatasetRegistry:
    """Loads and merges samples coming from structurally different sources."""

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or load_config()
        self.loaders: Dict[str, BaseDatasetLoader] = {
            name: cls(self.cfg) for name, cls in LOADERS.items()}

    def available(self) -> List[Dict[str, object]]:
        return [loader.describe() for loader in self.loaders.values()]

    def load_source(self, name: str, limit: Optional[int] = None,
                    phishing_only: Optional[bool] = None) -> List[Sample]:
        loader = self.loaders.get(name)
        if loader is None:
            raise KeyError(f"Unknown dataset source '{name}'. "
                           f"Known: {sorted(self.loaders)}")
        return loader.load(limit=limit, phishing_only=phishing_only) if loader.enabled else []

    def load_all(self, limit_per_source: Optional[int] = None,
                 sources: Optional[List[str]] = None,
                 phishing_only: Optional[bool] = None) -> List[Sample]:
        merged: List[Sample] = []
        for name in (sources or list(self.loaders)):
            try:
                merged.extend(self.load_source(name, limit_per_source, phishing_only))
            except Exception as exc:
                log.error("Failed to load source %s: %s", name, exc)
        return merged

    def sample(self, n: int, sources: Optional[List[str]] = None,
               seed: Optional[int] = None, phishing_only: bool = True,
               stratify: bool = True) -> List[Sample]:
        """Random subset, optionally balanced across the two sources."""
        rng = random.Random(seed if seed is not None
                            else self.cfg.get("evaluation.random_seed", 42))
        wanted = sources or list(self.loaders)
        if not stratify:
            pool = self.load_all(sources=wanted, phishing_only=phishing_only)
            rng.shuffle(pool)
            return pool[:n]

        per_source = max(1, n // max(1, len(wanted)))
        picked: List[Sample] = []
        leftovers: List[Sample] = []
        for name in wanted:
            pool = self.load_source(name, phishing_only=phishing_only)
            rng.shuffle(pool)
            picked.extend(pool[:per_source])
            leftovers.extend(pool[per_source:])
        rng.shuffle(leftovers)
        picked.extend(leftovers[: max(0, n - len(picked))])
        rng.shuffle(picked)
        return picked[:n]

    def by_ids(self, sample_ids: List[str]) -> List[Sample]:
        index = {s.sample_id: s for s in self.load_all(phishing_only=False)}
        return [index[sid] for sid in sample_ids if sid in index]

    def stats(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for name, loader in self.loaders.items():
            if not loader.exists():
                out[name] = {"total": 0, "phishing": 0, "benign": 0}
                continue
            items = loader.load(phishing_only=False)
            out[name] = {"total": len(items),
                         "phishing": sum(1 for s in items if s.is_phishing),
                         "benign": sum(1 for s in items if not s.is_phishing)}
        return out
