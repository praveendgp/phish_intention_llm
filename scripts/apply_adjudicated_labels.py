import argparse
from pathlib import Path
import pandas as pd
LABELS=["credential_theft","financial_fraud","malware_distribution","personal_information_harvesting"]
p=argparse.ArgumentParser();p.add_argument("--manifest",default="data/processed/manifest.csv");p.add_argument("--adjudicated",default="data/annotations/final_adjudicated.csv");p.add_argument("--backup",default="data/processed/manifest_before_adjudication.csv");a=p.parse_args();manifest=pd.read_csv(a.manifest).fillna("");final=pd.read_csv(a.adjudicated).fillna("");required={"sample_id",*LABELS};missing=required-set(final.columns)
if missing:raise SystemExit("Missing adjudication columns: "+", ".join(sorted(missing)))
Path(a.backup).parent.mkdir(parents=True,exist_ok=True);manifest.to_csv(a.backup,index=False);final=final.set_index("sample_id")
for i,row in manifest.iterrows():
    sid=row.sample_id
    if sid not in final.index:continue
    for label in LABELS:manifest.at[i,label]=int(final.at[sid,label])
    manifest.at[i,"annotation_status"]="adjudicated";manifest.at[i,"annotator"]="human_adjudicator"
manifest.to_csv(a.manifest,index=False);print(f"Updated {a.manifest}; backup: {a.backup}")
