from pathlib import Path
from .schema import VisionEvidence,Candidate,AnalysisResult,INTENTS
from . import prompts
class PhishIntentionPipeline:
 def __init__(self,llm,kb,threshold=.55,top_k=3): self.llm,self.kb,self.threshold,self.top_k=llm,kb,threshold,top_k
 def run(self,image_path,mode="gated",sample_id=None):
  sid=sample_id or Path(image_path).stem; agents=[]
  if mode=="single":
   out=self.llm.ask(prompts.baseline(),image_path); labels=[x for x in out.get("labels",[]) if x in INTENTS]
   return AnalysisResult(sample_id=sid,labels=labels,confidence=float(out.get("confidence",0)),evidence=out.get("evidence",{}),candidates=[],agents_invoked=["single_agent"],evidence_consistency=1.0 if labels else 0.0,trace={"mode":mode})
  ev=VisionEvidence.model_validate(self.llm.ask(prompts.vision(),image_path)); agents.append("vision_analysis")
  q=ev.visible_text+" "+" ".join(ev.ui_elements+ev.visual_cues)
  common=self.kb.search(q,4); cand_raw=self.llm.ask(prompts.classify(ev.model_dump(),common)); agents+= ["context_enrichment","classification"]
  cands=[]
  for x in cand_raw.get("candidates",[]):
   try:cands.append(Candidate.model_validate(x))
   except Exception:pass
  cands=sorted(cands,key=lambda x:x.confidence,reverse=True)[:self.top_k]
  active=INTENTS if mode=="always" else [c.intent for c in cands if c.confidence>=self.threshold]
  if not active and cands:active=[cands[0].intent]
  reports={}
  for label in active:
   knowledge=self.kb.search(q,4,label); reports[label]=self.llm.ask(prompts.specialist(label,ev.model_dump(),knowledge)); agents.append(label+"_specialist")
  val=self.llm.ask(prompts.validate(ev.model_dump(),[x.model_dump() for x in cands],reports)); agents.append("validator")
  confidence=float(val.get("confidence",0))
  if confidence<self.threshold and mode=="gated":
   for label in INTENTS:
    if label not in reports:
     reports[label]=self.llm.ask(prompts.specialist(label,ev.model_dump(),self.kb.search(q,4,label))); agents.append(label+"_specialist_feedback")
   val=self.llm.ask(prompts.validate(ev.model_dump(),[x.model_dump() for x in cands],reports)); confidence=float(val.get("confidence",0)); agents.append("validator_feedback")
  labels=[x for x in val.get("labels",[]) if x in INTENTS]
  return AnalysisResult(sample_id=sid,labels=labels,confidence=confidence,evidence=val.get("evidence",{}),candidates=cands,agents_invoked=agents,evidence_consistency=float(val.get("evidence_consistency",0)),trace={"mode":mode,"vision":ev.model_dump(),"retrieved_common":common,"specialists":reports})
