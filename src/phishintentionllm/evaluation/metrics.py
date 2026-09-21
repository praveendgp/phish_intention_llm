"""Evaluation metrics from Section 5.3 of the base paper.

    Accuracy            - exact match between predicted and reference label sets
    Precision_micro     - sum(TP_c) / sum(TP_c + FP_c)      (Eq. 4)
    Recall_micro        - sum(TP_c) / sum(TP_c + FN_c)      (Eq. 5)
    F1_micro            - harmonic mean                     (Eq. 6)
    Accuracy_micro      - (TP+TN) / (TP+TN+FP+FN)
    Acc_comp(k)         - partial-match accuracy by complexity, with
                          t_k = 1 for k in {1, 2} and t_k = 2 for k = 3  (Eq. 7-8)
    Per-class P/R/F1/Accuracy                                (Eq. 9-10)

"Reference" here is the manifest produced by the annotator ensemble (or the
human-reviewed ground truth, depending on evaluation.reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

from ..schemas import Intention

CATEGORIES: List[str] = [i.value for i in Intention.all()]
DEFAULT_THRESHOLDS: Dict[int, int] = {1: 1, 2: 1, 3: 2, 4: 2}


def _norm(labels: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for label in labels or []:
        category = Intention.coerce(label)
        if category:
            out.add(category.value)
    return out


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
                "accuracy": round(self.accuracy, 4)}


@dataclass
class EvaluationReport:
    n_samples: int = 0
    overall_accuracy: float = 0.0
    precision_micro: float = 0.0
    recall_micro: float = 0.0
    f1_micro: float = 0.0
    accuracy_micro: float = 0.0
    acc_by_complexity: Dict[str, float] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    support: Dict[str, int] = field(default_factory=dict)
    totals: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"n_samples": self.n_samples,
                "overall_accuracy": round(self.overall_accuracy, 4),
                "precision_micro": round(self.precision_micro, 4),
                "recall_micro": round(self.recall_micro, 4),
                "f1_micro": round(self.f1_micro, 4),
                "accuracy_micro": round(self.accuracy_micro, 4),
                "acc_by_complexity": {k: round(v, 4)
                                      for k, v in self.acc_by_complexity.items()},
                "per_class": self.per_class, "support": self.support,
                "totals": self.totals}


def evaluate(y_true: Sequence[Iterable[str]], y_pred: Sequence[Iterable[str]],
             thresholds: Dict[int, int] | None = None,
             categories: Sequence[str] | None = None) -> EvaluationReport:
    """Compute the full metric suite for multi-label intention predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    cats = list(categories or CATEGORIES)
    thresholds = {int(k): int(v) for k, v in (thresholds or DEFAULT_THRESHOLDS).items()}

    truths = [_norm(t) for t in y_true]
    preds = [_norm(p) for p in y_pred]
    n = len(truths)

    counts: Dict[str, ConfusionCounts] = {c: ConfusionCounts() for c in cats}
    exact_matches = 0
    complexity: Dict[int, List[bool]] = {}

    for truth, pred in zip(truths, preds):
        if truth == pred:
            exact_matches += 1
        for category in cats:
            in_t, in_p = category in truth, category in pred
            bucket = counts[category]
            if in_t and in_p:
                bucket.tp += 1
            elif in_p:
                bucket.fp += 1
            elif in_t:
                bucket.fn += 1
            else:
                bucket.tn += 1
        k = len(truth)
        if k:
            t_k = thresholds.get(k, max(1, k - 1))
            complexity.setdefault(k, []).append(len(truth & pred) >= t_k)

    tp = sum(c.tp for c in counts.values())
    fp = sum(c.fp for c in counts.values())
    fn = sum(c.fn for c in counts.values())
    tn = sum(c.tn for c in counts.values())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return EvaluationReport(
        n_samples=n,
        overall_accuracy=exact_matches / n if n else 0.0,
        precision_micro=precision, recall_micro=recall, f1_micro=f1,
        accuracy_micro=(tp + tn) / (tp + tn + fp + fn) if n else 0.0,
        acc_by_complexity={f"{k} intention(s)": sum(v) / len(v)
                           for k, v in sorted(complexity.items()) if v},
        per_class={c: counts[c].to_dict() for c in cats},
        support={c: sum(1 for t in truths if c in t) for c in cats},
        totals={"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "exact_matches": exact_matches},
    )


def evaluate_binary(y_true: Sequence[Iterable[str]], y_pred: Sequence[Iterable[str]],
                    category: str = Intention.CREDENTIAL_THEFT.value
                    ) -> Dict[str, float]:
    """Single-category benchmark (the PhishIntention comparison of the paper)."""
    counts = ConfusionCounts()
    for truth, pred in zip([_norm(t) for t in y_true], [_norm(p) for p in y_pred]):
        in_t, in_p = category in truth, category in pred
        if in_t and in_p:
            counts.tp += 1
        elif in_p:
            counts.fp += 1
        elif in_t:
            counts.fn += 1
        else:
            counts.tn += 1
    payload = counts.to_dict()
    payload["category"] = category
    payload["n_samples"] = len(list(y_true))
    return payload


def agreement_rate(y_true: Sequence[Iterable[str]],
                   y_pred: Sequence[Iterable[str]]) -> Dict[str, float]:
    """Set-level agreement between framework predictions and the reference."""
    exact = partial = disjoint = 0
    jaccard_total = 0.0
    for truth, pred in zip([_norm(t) for t in y_true], [_norm(p) for p in y_pred]):
        union = truth | pred
        inter = truth & pred
        jaccard_total += (len(inter) / len(union)) if union else 1.0
        if truth == pred:
            exact += 1
        elif inter:
            partial += 1
        else:
            disjoint += 1
    n = max(1, len(list(y_true)))
    return {"exact_match": round(exact / n, 4),
            "partial_match": round(partial / n, 4),
            "no_overlap": round(disjoint / n, 4),
            "mean_jaccard": round(jaccard_total / n, 4)}


def co_occurrence(label_sets: Sequence[Iterable[str]]) -> Dict[str, int]:
    """Frequency of two- and three-intention combinations (Fig. 7b analogue)."""
    from itertools import combinations

    out: Dict[str, int] = {}
    for labels in label_sets:
        normalised = sorted(_norm(labels))
        for size in (2, 3):
            if len(normalised) < size:
                continue
            for combo in combinations(normalised, size):
                key = " + ".join(Intention.short(c) for c in combo)
                out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def sector_matrix(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    """Sector x intention frequency matrix (Fig. 7a analogue)."""
    matrix: Dict[str, Dict[str, int]] = {}
    for row in rows:
        sector = str(row.get("sector", "other") or "other")
        bucket = matrix.setdefault(sector, {c: 0 for c in CATEGORIES})
        for label in _norm(row.get("labels", [])):
            bucket[label] = bucket.get(label, 0) + 1
    return matrix
