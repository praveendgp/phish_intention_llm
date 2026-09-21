"""Loader for the Phish-IRIS dataset (Dalgic et al., ISMSIT 2018).

On-disk layout
--------------
    phishIris/
      train/<brand>/*.png
      val/<brand>/*.png

Every image is a phishing screenshot; the directory name is the impersonated
brand ("other" is a catch-all class). Brand is carried as metadata and used as
a weak sector prior.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..schemas import Sample
from ..utils.logging import get_logger
from .base import BaseDatasetLoader

log = get_logger(__name__)

BRAND_SECTOR: Dict[str, str] = {
    "apple": "online/cloud service",
    "adobe": "online/cloud service",
    "microsoft": "online/cloud service",
    "dropbox": "online/cloud service",
    "amazon": "e-commerce",
    "alibaba": "e-commerce",
    "ebay": "e-commerce",
    "chase": "financial",
    "boa": "financial",
    "wellsfargo": "financial",
    "paypal": "payment services",
    "facebook": "social networking",
    "linkedin": "social networking",
    "yahoo": "email provider",
    "dhl": "delivery & couriers",
    "other": "other",
}


class PhishIrisLoader(BaseDatasetLoader):
    source_name = "phishiris"

    def load(self, limit: Optional[int] = None,
             phishing_only: Optional[bool] = None) -> List[Sample]:
        if not self.exists():
            log.warning("Phish-IRIS dataset not found at %s", self.root)
            return []

        samples: List[Sample] = []
        for split in self.settings.get("splits", ["train", "val"]):
            split_root = self.root / split
            if not split_root.exists():
                continue
            for brand_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
                brand = brand_dir.name.lower()
                for image in self._images_in(brand_dir):
                    samples.append(Sample(
                        sample_id=f"phishiris::{split}::{brand}::{image.stem}",
                        source=self.source_name,
                        screenshot_path=str(image),
                        is_phishing=True,     # every Phish-IRIS sample is phishing
                        brand=brand,
                        split=split,
                        record_dir=str(brand_dir),
                        extra={"brand": brand,
                               "sector_hint": BRAND_SECTOR.get(brand, "other"),
                               "file_name": image.name},
                    ))
                    if limit and len(samples) >= limit:
                        return samples
        log.info("Phish-IRIS: loaded %s samples from %s", len(samples), self.root)
        return samples
