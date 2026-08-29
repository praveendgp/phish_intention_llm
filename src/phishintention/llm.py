import base64, json, re, requests
from pathlib import Path
from PIL import Image
from io import BytesIO
class OllamaClient:
    def __init__(self, base_url, model, max_side=1600): self.base_url,self.model,self.max_side=base_url.rstrip('/'),model,max_side
    def _image_b64(self,path):
        im=Image.open(path).convert('RGB'); im.thumbnail((self.max_side,self.max_side)); b=BytesIO(); im.save(b,'JPEG',quality=85); return base64.b64encode(b.getvalue()).decode()
    def ask(self,prompt,image_path=None):
        payload={"model":self.model,"prompt":prompt,"stream":False,"format":"json"}
        if image_path: payload["images"]=[self._image_b64(image_path)]
        r=requests.post(self.base_url+"/api/generate",json=payload,timeout=300); r.raise_for_status()
        raw=r.json()["response"]
        try:return json.loads(raw)
        except json.JSONDecodeError:
            m=re.search(r'\{.*\}',raw,re.S)
            if not m: raise ValueError("Model did not return JSON: "+raw[:300])
            return json.loads(m.group())
class MockClient:
    def ask(self,prompt,image_path=None):
        p=prompt.lower()
        if 'vision analyst' in p:return {"visible_text":"Sign in Email Password","ui_elements":["login form","password field"],"visual_cues":["brand logo"],"brand_signals":["demo"]}
        if 'initial multi-label' in p:return {"candidates":[{"intent":"credential_theft","confidence":0.86,"evidence":["password field"]}]}
        if 'validator' in p:return {"labels":["credential_theft"],"confidence":0.87,"evidence":{"credential_theft":["login form requests password"]},"evidence_consistency":1.0}
        if 'specialist' in p:return {"confidence":0.88,"evidence":["login form requests password"],"supported":True}
        return {"labels":["credential_theft"],"confidence":0.8,"evidence":{"credential_theft":["password field"]}}
