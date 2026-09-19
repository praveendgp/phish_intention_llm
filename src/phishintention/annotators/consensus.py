from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS = ["credential_theft", "financial_fraud", "malware_distribution", "personal_information_harvesting"]

def load_jsonl(path: str | Path) -> dict[str, dict]:
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line); rows[item["sample_id"]] = item
    return rows

def annotation_label(item: dict, label: str) -> int:
    return int(bool(item["annotation"][label]["present"]))

def compare(openai_path: str | Path, gemini_path: str | Path) -> tuple[pd.DataFrame, dict]:
    left, right = load_jsonl(openai_path), load_jsonl(gemini_path)
    shared = sorted(set(left) & set(right)); rows=[]
    for sample_id in shared:
        a,b=left[sample_id],right[sample_id]; row={"sample_id":sample_id,"openai_quality":a["annotation"]["image_quality"],"gemini_quality":b["annotation"]["image_quality"]}; all_match=True
        for label in LABELS:
            av,bv=annotation_label(a,label),annotation_label(b,label);row[f"openai_{label}"]=av;row[f"gemini_{label}"]=bv;row[f"agree_{label}"]=av==bv;all_match &= av==bv
        row["exact_labelset_agreement"]=all_match;row["adjudication_required"]=(not all_match) or row["openai_quality"]!="usable" or row["gemini_quality"]!="usable";rows.append(row)
    frame=pd.DataFrame(rows);metrics={"shared_samples":len(frame),"exact_labelset_agreement":None if frame.empty else float(frame.exact_labelset_agreement.mean()),"per_label":{}}
    for label in LABELS:
        if frame.empty: metrics["per_label"][label]={"raw_agreement":None,"cohen_kappa":None};continue
        a=frame[f"openai_{label}"].tolist();b=frame[f"gemini_{label}"].tolist();metrics["per_label"][label]={"raw_agreement":float((frame[f"agree_{label}"]).mean()),"cohen_kappa":float(cohen_kappa_score(a,b)) if len(set(a+b))>1 else None}
    return frame, metrics
