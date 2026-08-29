import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phishintention.config import Settings
from phishintention.llm import OllamaClient,MockClient
from phishintention.retrieval import KnowledgeBase
from phishintention.pipeline import PhishIntentionPipeline
p=argparse.ArgumentParser();p.add_argument("--image",required=True);p.add_argument("--mode",choices=["single","always","gated"],default="gated");p.add_argument("--provider",choices=["ollama","mock"],default="ollama");a=p.parse_args();s=Settings();llm=MockClient() if a.provider=="mock" else OllamaClient(s.base_url,s.model,s.max_image_side);pipe=PhishIntentionPipeline(llm,KnowledgeBase("knowledge"),s.threshold,s.top_k);print(pipe.run(a.image,a.mode).model_dump_json(indent=2))
