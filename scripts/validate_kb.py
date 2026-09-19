from pathlib import Path
from collections import Counter
import json, sys
root=Path(__file__).resolve().parents[1]/'knowledge'
required={'id','category','knowledge_type','text','source','retrieval_terms','polarity'}
valid_categories={'common','credential_theft','financial_fraud','malware_distribution','personal_information_harvesting'}
ids=set(); errors=[]; counts=Counter()
for path in sorted(root.glob('*.json')):
    try: entries=json.loads(path.read_text(encoding='utf-8'))
    except Exception as e: errors.append(f'{path}: invalid JSON: {e}'); continue
    if not isinstance(entries,list): errors.append(f'{path}: root must be a list'); continue
    for n,e in enumerate(entries,1):
        if not isinstance(e,dict): errors.append(f'{path}:{n}: entry is not object'); continue
        missing=required-set(e)
        if missing: errors.append(f'{path}:{n}: missing {sorted(missing)}')
        eid=e.get('id')
        if eid in ids: errors.append(f'{path}:{n}: duplicate id {eid}')
        ids.add(eid)
        cat=e.get('category')
        if cat not in valid_categories: errors.append(f'{path}:{n}: invalid category {cat}')
        counts[cat]+=1
        if not str(e.get('text','')).strip(): errors.append(f'{path}:{n}: empty text')
        if not isinstance(e.get('retrieval_terms'),list) or not e.get('retrieval_terms'): errors.append(f'{path}:{n}: retrieval_terms must be nonempty list')
print('Entry counts:')
for k,v in sorted(counts.items()): print(f'  {k}: {v}')
print(f'Total: {sum(counts.values())}')
if errors:
    print('\nErrors:',file=sys.stderr)
    for e in errors: print(' -',e,file=sys.stderr)
    raise SystemExit(1)
print('\nKB validation passed.')
