import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class KnowledgeBase:
 def __init__(self,root):
  docs=[]
  for p in Path(root).glob("*.json"):
   data=json.loads(p.read_text())
   for x in data: docs.append({**x,"source":p.name})
  self.docs=docs; self.vec=TfidfVectorizer(ngram_range=(1,2),stop_words="english")
  self.mat=self.vec.fit_transform([d["text"] for d in docs]) if docs else None
 def search(self,query,k=4,category=None):
  pool=[i for i,d in enumerate(self.docs) if not category or d.get("category") in (category,"common")]
  if not pool:return []
  q=self.vec.transform([query]); scores=cosine_similarity(q,self.mat[pool]).ravel()
  return [{**self.docs[pool[i]],"score":float(scores[i])} for i in scores.argsort()[::-1][:k]]
