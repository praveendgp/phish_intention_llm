"""Configuration loader for PhishIntentionLLM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Roles that make up each stage.
MANIFEST_ROLES = ["annotator_a", "annotator_b",
                  "manifest_tiebreaker", "manifest_finalizer"]
FRAMEWORK_ROLES = ["vision", "context", "classifier", "specialist", "validator"]
ALL_ROLES = MANIFEST_ROLES + FRAMEWORK_ROLES


@dataclass
class ModelSpec:
    role: str
    model: str
    vision: bool = True
    temperature: float = 0.0
    top_p: float = 0.9
    label: str = ""

    @classmethod
    def from_dict(cls, role: str, data: Dict[str, Any]) -> "ModelSpec":
        return cls(
            role=role,
            model=data["model"],
            vision=bool(data.get("vision", True)),
            temperature=float(data.get("temperature", 0.0)),
            top_p=float(data.get("top_p", 0.9)),
            label=data.get("label", data["model"]),
        )


class Config:
    """Dict wrapper with dotted access and repo-relative path resolution."""

    def __init__(self, data: Dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = path or DEFAULT_CONFIG_PATH

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    def resolve(self, dotted: str, default: str = "") -> Path:
        value = self.get(dotted, default)
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    # -- models ---------------------------------------------------------
    def model(self, role: str) -> ModelSpec:
        spec = self.get(f"models.{role}")
        if not spec:
            raise KeyError(f"No model configured for role '{role}'. "
                           f"Known roles: {ALL_ROLES}")
        return ModelSpec.from_dict(role, spec)

    def manifest_models(self) -> List[ModelSpec]:
        return [self.model(r) for r in MANIFEST_ROLES if self.get(f"models.{r}")]

    def framework_models(self) -> List[ModelSpec]:
        return [self.model(r) for r in FRAMEWORK_ROLES if self.get(f"models.{r}")]

    def all_models(self) -> List[ModelSpec]:
        return [self.model(r) for r in ALL_ROLES if self.get(f"models.{r}")]

    @property
    def vision_for_all_agents(self) -> bool:
        return bool(self.get("framework.vision_for_all_agents", True))

    # -- misc -----------------------------------------------------------
    @property
    def categories(self) -> List[str]:
        return list(self.get("categories", []))

    @property
    def sectors(self) -> List[str]:
        return list(self.get("sectors", ["other"]))

    @property
    def ollama_host(self) -> str:
        return os.environ.get("OLLAMA_HOST") or self.get(
            "ollama.host", "http://localhost:11434")

    def ensure_dirs(self) -> None:
        for key in ("storage.results_dir", "storage.runs_dir",
                    "storage.kb_dir", "storage.manifest_runs_dir"):
            self.resolve(key).mkdir(parents=True, exist_ok=True)
        for key in ("storage.manifest", "storage.predictions",
                    "storage.manual_annotations", "storage.ground_truth"):
            self.resolve(key).parent.mkdir(parents=True, exist_ok=True)


_CACHE: Dict[str, Config] = {}


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else Path(
        os.environ.get("PHISHINTENTION_CONFIG", DEFAULT_CONFIG_PATH))
    key = str(cfg_path.resolve())
    if key in _CACHE:
        return _CACHE[key]
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data, cfg_path)
    _CACHE[key] = cfg
    return cfg
