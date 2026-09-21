"""Screenshot loading / encoding helpers for the vision-language agents."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image
    _PIL = True
except Exception:  # pragma: no cover
    _PIL = False

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def load_image_b64(path: str | Path, max_edge: int = 1280) -> str:
    """Return a base64 PNG string, down-scaled to `max_edge` on the long side.

    Every agent in the framework is handed this same encoded screenshot, so the
    image is prepared once per sample and reused across all vision calls.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Screenshot not found: {p}")
    if not _PIL:
        return base64.b64encode(p.read_bytes()).decode("ascii")

    with Image.open(p) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_edge:
            scale = max_edge / float(max(w, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_size(path: str | Path) -> Optional[Tuple[int, int]]:
    if not _PIL:
        return None
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None
