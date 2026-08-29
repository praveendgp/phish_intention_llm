import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--limit",type=int);a=p.parse_args()
for mode in ["single","always","gated"]:
 cmd=[sys.executable,"scripts/evaluate.py","--manifest",a.manifest,"--mode",mode]+(["--limit",str(a.limit)] if a.limit else []);subprocess.run(cmd,check=True)
