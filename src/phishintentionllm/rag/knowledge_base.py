"""Dual-layer knowledge architecture (Definition 1 of the base paper).

    K_B = {P_c, P_v, P_t}                      -> basic threat patterns
    K_c = {F_c, D_c}, D_c = {T_c, M_c, I_c}    -> specialist knowledge per category
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Config, load_config


@dataclass
class Passage:
    """An atomic retrievable knowledge unit."""

    text: str
    repository: str            # "basic" | "specialist"
    category: Optional[str]    # None for the basic repository
    facet: str

    def to_dict(self) -> Dict[str, str]:
        return {"text": self.text, "repository": self.repository,
                "category": self.category or "-", "facet": self.facet}


FACET_LABELS = {
    "common_patterns": "P_c (common phishing patterns)",
    "visual_deception": "P_v (visual deception techniques)",
    "text_manipulation": "P_t (text-based manipulation)",
    "url_red_flags": "P_u (URL red flags)",
    "primary_features": "F_c (primary features)",
    "common_targets": "T_c (common targets)",
    "techniques": "M_c (specialised techniques)",
    "indicators": "I_c (distinctive indicators)",
    "counter_indicators": "N_c (counter indicators)",
    "co_occurrence": "co-occurrence prior",
}


@dataclass
class KnowledgeBase:
    basic: Dict = field(default_factory=dict)
    specialist: Dict = field(default_factory=dict)
    passages: List[Passage] = field(default_factory=list)

    @classmethod
    def load(cls, config: Optional[Config] = None) -> "KnowledgeBase":
        cfg = config or load_config()
        kb_dir: Path = cfg.resolve("storage.kb_dir", "./data/knowledge_base")
        with open(kb_dir / "basic_patterns.json", "r", encoding="utf-8") as fh:
            basic = json.load(fh)
        with open(kb_dir / "specialist_kb.json", "r", encoding="utf-8") as fh:
            specialist = json.load(fh)
        kb = cls(basic=basic, specialist=specialist)
        kb.passages = kb._build_passages()
        return kb

    def _build_passages(self) -> List[Passage]:
        out: List[Passage] = []
        for facet in ("common_patterns", "visual_deception",
                      "text_manipulation", "url_red_flags"):
            for item in self.basic.get(facet, []):
                out.append(Passage(item, "basic", None, FACET_LABELS[facet]))

        for category, block in self.specialist.get("categories", {}).items():
            out.append(Passage(block.get("definition", ""), "specialist",
                               category, "definition"))
            for facet in ("primary_features", "common_targets", "techniques",
                          "indicators", "counter_indicators"):
                for item in block.get(facet, []):
                    out.append(Passage(item, "specialist", category,
                                       FACET_LABELS[facet]))

        for note in self.specialist.get("co_occurrence_notes", []):
            out.append(Passage(note, "specialist", None,
                               FACET_LABELS["co_occurrence"]))
        return [p for p in out if p.text]

    # RETRIEVEPATTERNS(K_B) - Algorithm line 6
    def retrieve_patterns(self, limit_per_facet: int = 6) -> List[str]:
        out: List[str] = []
        for facet in ("common_patterns", "visual_deception",
                      "text_manipulation", "url_red_flags"):
            for item in self.basic.get(facet, [])[:limit_per_facet]:
                out.append(f"[{FACET_LABELS[facet]}] {item}")
        return out

    # RETRIEVECATEGORYFEATURES(K_C) - Algorithm line 10
    def retrieve_category_features(self) -> Dict[str, List[str]]:
        return {name: block.get("primary_features", [])
                for name, block in self.specialist.get("categories", {}).items()}

    # RETRIEVESPECIALISTKNOWLEDGE(K_C, c) - Algorithm lines 15 / 24
    def retrieve_specialist_knowledge(self, category: str) -> Dict[str, object]:
        block = self.specialist.get("categories", {}).get(category, {})
        return {
            "category": category,
            "definition": block.get("definition", ""),
            "F_c_primary_features": block.get("primary_features", []),
            "T_c_common_targets": block.get("common_targets", []),
            "M_c_techniques": block.get("techniques", []),
            "I_c_indicators": block.get("indicators", []),
            "counter_indicators": block.get("counter_indicators", []),
        }

    def sector_cues(self) -> Dict[str, List[str]]:
        return self.basic.get("sector_cues", {})

    def co_occurrence_notes(self) -> List[str]:
        return self.specialist.get("co_occurrence_notes", [])

    def stats(self) -> Dict[str, int]:
        return {
            "passages": len(self.passages),
            "basic_passages": sum(1 for p in self.passages if p.repository == "basic"),
            "specialist_passages": sum(1 for p in self.passages
                                       if p.repository == "specialist"),
            "categories": len(self.specialist.get("categories", {})),
        }


@lru_cache(maxsize=4)
def get_knowledge_base(config_path: Optional[str] = None) -> KnowledgeBase:
    return KnowledgeBase.load(load_config(config_path))
