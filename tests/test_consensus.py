import json
from phishintention.annotators.consensus import compare

def record(sid,ct=False,pih=False):
 d={"sample_id":sid,"annotation":{"image_quality":"usable"}}
 for label,value in {"credential_theft":ct,"financial_fraud":False,"malware_distribution":False,"personal_information_harvesting":pih}.items():d["annotation"][label]={"present":value,"confidence":.9,"evidence":[]}
 return d

def test_agreement(tmp_path):
 a=tmp_path/"a.jsonl";b=tmp_path/"b.jsonl";a.write_text(json.dumps(record("1",True))+"\n");b.write_text(json.dumps(record("1",True))+"\n");frame,metrics=compare(a,b);assert bool(frame.iloc[0].exact_labelset_agreement);assert metrics["exact_labelset_agreement"]==1.0
