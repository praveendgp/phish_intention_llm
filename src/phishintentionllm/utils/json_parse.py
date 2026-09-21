"""Tolerant JSON extraction from VLM output.

Small open-weight vision models frequently wrap JSON in prose or markdown
fences, emit trailing commas or single quotes. These helpers recover a usable
dict instead of failing the whole pipeline.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def _balanced_slice(text: str, opener: str = "{", closer: str = "}") -> Optional[str]:
    start = text.find(opener)
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _repair(candidate: str) -> str:
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", fixed)


def parse_json(text: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the first JSON object found in `text`, or `default`."""
    if not text:
        return dict(default or {})
    cleaned = _strip_fences(text)
    for candidate in (cleaned, _balanced_slice(cleaned) or ""):
        if not candidate:
            continue
        for attempt in (candidate, _repair(candidate)):
            try:
                value = json.loads(attempt)
                if isinstance(value, dict):
                    return value
                if isinstance(value, list):
                    return {"items": value}
            except Exception:
                pass
            try:
                value = ast.literal_eval(attempt)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    return dict(default or {})


def as_list(value: Any) -> List[str]:
    """Coerce whatever the model returned into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n;|]+|,(?![^(]*\))", value)
        return [p.strip(" -*\t") for p in parts if p and p.strip(" -*\t")]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if isinstance(item, (dict, list, tuple, set)):
                out.extend(as_list(item))
            elif item is not None and str(item).strip():
                out.append(str(item).strip())
        return out
    return [str(value)]


def as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a score to a 0-1 float ('85%', '0.85', 8.5/10, 85 all work)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        if isinstance(value, str):
            number = float(re.findall(r"-?\d+\.?\d*", value.replace("%", ""))[0])
            if "%" in value:
                number /= 100.0
        else:
            number = float(value)
    except Exception:
        return default
    if number > 1.0:
        number = number / 100.0 if number > 10 else number / 10.0
    return max(0.0, min(1.0, round(number, 4)))


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "yes", "y", "1", "keep", "confirmed"}
