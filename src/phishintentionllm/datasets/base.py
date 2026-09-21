"""Common interface for the heterogeneous dataset sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..config import Config, load_config
from ..schemas import Sample
from ..utils.image import IMAGE_SUFFIXES


class BaseDatasetLoader(ABC):
    """Every source normalises its own layout into a list of `Sample`."""

    source_name: str = "base"

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or load_config()
        self.settings: Dict = self.cfg.get(f"datasets.{self.source_name}", {}) or {}

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("enabled", True))

    @property
    def root(self) -> Path:
        return self.cfg.resolve(f"datasets.{self.source_name}.path")

    def exists(self) -> bool:
        return self.root.exists()

    @abstractmethod
    def load(self, limit: Optional[int] = None,
             phishing_only: Optional[bool] = None) -> List[Sample]:
        """Return normalised samples for this source."""

    @staticmethod
    def _images_in(directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        return sorted(p for p in directory.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    @staticmethod
    def _pick_primary(images: Iterable[Path]) -> Optional[Path]:
        """Prefer a full-page/desktop shot when several are present."""
        images = list(images)
        if not images:
            return None
        for key in ("shot", "screenshot", "full", "desktop", "page", "home"):
            for img in images:
                if key in img.stem.lower():
                    return img
        return max(images, key=lambda p: p.stat().st_size)

    def describe(self) -> Dict[str, object]:
        return {"source": self.source_name, "path": str(self.root),
                "exists": self.exists(), "enabled": self.enabled}
