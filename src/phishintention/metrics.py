import numpy as np
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
def accuracy_by_complexity(y_true,y_pred):
 out={}
 for k,t in [(1,1),(2,1),(3,2)]:
  idx=np.where(y_true.sum(axis=1)==k)[0]
  out[str(k)]=None if len(idx)==0 else float(((y_true[idx]&y_pred[idx]).sum(axis=1)>=t).mean())
 return out
def compute(y_true,y_pred):
 p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='micro',zero_division=0)
 pp,rr,ff,_=precision_recall_fscore_support(y_true,y_pred,average=None,zero_division=0)
 return {'micro_precision':float(p),'micro_recall':float(r),'micro_f1':float(f),'subset_accuracy':float(accuracy_score(y_true,y_pred)),'accuracy_by_complexity':accuracy_by_complexity(y_true,y_pred),'per_label':{'precision':pp.tolist(),'recall':rr.tolist(),'f1':ff.tolist()}}
