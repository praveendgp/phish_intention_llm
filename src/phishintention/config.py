from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()
@dataclass
class Settings:
    base_url:str=os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
    model:str=os.getenv("OLLAMA_MODEL","qwen2.5vl:3b")
    threshold:float=float(os.getenv("CONFIDENCE_THRESHOLD","0.55"))
    top_k:int=int(os.getenv("TOP_K","3"))
    max_image_side:int=int(os.getenv("MAX_IMAGE_SIDE","1600"))
