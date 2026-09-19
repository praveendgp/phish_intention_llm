from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image

def normalised_image_bytes(path: str | Path, max_side: int = 1600) -> bytes:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue()

def image_data_uri(path: str | Path, max_side: int = 1600) -> str:
    encoded = base64.b64encode(normalised_image_bytes(path, max_side)).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
