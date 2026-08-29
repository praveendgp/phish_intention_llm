import argparse,sys,json
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phishintention.schema import INTENTS
from phishintention.metrics import compute
from phishintention.config import Settings
from phishintention.llm import OllamaClient
from phishintention.retrieval import KnowledgeBase
from phishintention.pipeline import PhishIntentionPipeline
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--mode',default='gated',choices=['single','always','gated']);p.add_argument('--limit',type=int);a=p.parse_args();d=pd.read_csv(a.manifest).fillna('');d=d[d.annotation_status.isin(['labelled','adjudicated'])];d=d.head(a.limit) if a.limit else d
if d.empty: raise SystemExit('No labelled/adjudicated rows. Annotate first.')
s=Settings();pipe=PhishIntentionPipeline(OllamaClient(s.base_url,s.model,s.max_image_side),KnowledgeBase('knowledge'),s.threshold,s.top_k);out=Path('outputs');out.mkdir(exist_ok=True);pred=[]
log=out/f'predictions_{a.mode}.jsonl';done=set()
if log.exists():
 for line in log.read_text().splitlines():
  x=json.loads(line);done.add(x['sample_id']);pred.append(x)
with log.open('a') as f:
 for _,r in d.iterrows():
  if r.sample_id in done:continue
  x=pipe.run(r.image_path,a.mode,r.sample_id).model_dump();f.write(json.dumps(x)+'\n');f.flush();pred.append(x)
by={x['sample_id']:x for x in pred};d=d[d.sample_id.isin(by)];yt=np.array([[int(str(r[i]).lower() in ('1','true','yes')) for i in INTENTS] for _,r in d.iterrows()]);yp=np.array([[int(i in by[r.sample_id]['labels']) for i in INTENTS] for _,r in d.iterrows()]);m=compute(yt,yp);(out/f'metrics_{a.mode}.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
