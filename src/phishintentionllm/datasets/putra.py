"""Loader for the Putra phishing-website dataset (Zenodo 8041387).

On-disk layout
--------------
    putra/
      phishing/<record-id>/screenshots/*.png
                          /assets/*
      not-phishing/<record-id>/screenshots/*.png
                               /assets/*

`<record-id>` is a Mongo-style ObjectId. The `assets` folder may carry the URL
or HTML captured with the screenshot; when present it is harvested as metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..schemas import Sample
from ..utils.logging import get_logger
from .base import BaseDatasetLoader

log = get_logger(__name__)

_URL_HINT_FILES = ("url.txt", "meta.json", "info.json", "metadata.json", "url")


class PutraLoader(BaseDatasetLoader):
    source_name = "putra"

    def load(self, limit: Optional[int] = None,
             phishing_only: Optional[bool] = None) -> List[Sample]:
        if not self.exists():
            log.warning("Putra dataset not found at %s", self.root)
            return []

        only_phish = (self.settings.get("phishing_only", True)
                      if phishing_only is None else phishing_only)
        shot_dir = self.settings.get("screenshot_dir", "screenshots")
        asset_dir = self.settings.get("asset_dir", "assets")
        labels = self.settings.get("labels", ["phishing", "not-phishing"])

        samples: List[Sample] = []
        for label in labels:
            is_phishing = (label == "phishing")
            if only_phish and not is_phishing:
                continue
            label_root = self.root / label
            if not label_root.exists():
                continue

            for record in sorted(p for p in label_root.iterdir() if p.is_dir()):
                images = self._images_in(record / shot_dir) or self._images_in(record)
                primary = self._pick_primary(images)
                if primary is None:
                    continue
                assets_path = record / asset_dir
                samples.append(Sample(
                    sample_id=f"putra::{label}::{record.name}",
                    source=self.source_name,
                    screenshot_path=str(primary),
                    is_phishing=is_phishing,
                    brand=None,
                    split=label,
                    url=self._read_url(assets_path) or self._read_url(record),
                    record_dir=str(record),
                    assets=[str(p) for p in self._images_in(assets_path)],
                    extra={"record_id": record.name, "label_dir": label,
                           "n_screenshots": len(images),
                           "all_screenshots": [str(p) for p in images]},
                ))
                if limit and len(samples) >= limit:
                    return samples
        log.info("Putra: loaded %s samples from %s", len(samples), self.root)
        return samples

    @staticmethod
    def _read_url(folder: Path) -> Optional[str]:
        if not folder.exists():
            return None
        for name in _URL_HINT_FILES:
            candidate = folder / name
            if not candidate.exists() or not candidate.is_file():
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if candidate.suffix == ".json":
                try:
                    blob = json.loads(text)
                    for key in ("url", "URL", "page_url", "target_url"):
                        if isinstance(blob, dict) and blob.get(key):
                            return str(blob[key])[:500]
                except Exception:
                    pass
            elif text.startswith("http"):
                return text.splitlines()[0][:500]
        return None
