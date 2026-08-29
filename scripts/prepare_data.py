import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phishintention.data import build_manifest
p=argparse.ArgumentParser();p.add_argument("--phish-iris");p.add_argument("--putra");p.add_argument("--out",default="data/processed/manifest.csv");a=p.parse_args();df=build_manifest(a.phish_iris,a.putra,a.out);print(f"Wrote {len(df)} rows to {a.out}")
