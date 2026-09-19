import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from phishintention.annotators.consensus import compare
p=argparse.ArgumentParser();p.add_argument("--openai",default="data/annotations/openai_annotations.jsonl");p.add_argument("--gemini",default="data/annotations/gemini_annotations.jsonl");p.add_argument("--output-dir",default="data/annotations");a=p.parse_args();frame,metrics=compare(a.openai,a.gemini);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/"agreement.csv",index=False);frame[frame.adjudication_required].to_csv(out/"adjudication_queue.csv",index=False);(out/"agreement_metrics.json").write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2));print(f"Adjudication queue: {int(frame.adjudication_required.sum()) if not frame.empty else 0}")
