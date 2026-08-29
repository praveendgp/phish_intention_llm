import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
import streamlit as st, pandas as pd
from PIL import Image
from phishintention.schema import INTENTS
from phishintention.config import Settings
from phishintention.llm import OllamaClient,MockClient
from phishintention.retrieval import KnowledgeBase
from phishintention.pipeline import PhishIntentionPipeline
st.set_page_config(page_title='PhishIntentionLLM',layout='wide');st.title('PhishIntentionLLM Two-Dataset Reproduction');st.caption('Defensive analysis of released static screenshots only')
tab1,tab2=st.tabs(['Analyse screenshot','Annotate dataset'])
with tab1:
 u=st.file_uploader('Screenshot',type=['png','jpg','jpeg','webp']);mode=st.selectbox('Mode',['gated','always','single']);mock=st.checkbox('Mock model for smoke test')
 if u:
  tmp=Path('outputs/uploaded_'+u.name);tmp.parent.mkdir(exist_ok=True);tmp.write_bytes(u.getvalue());st.image(Image.open(tmp),width=600)
  if st.button('Analyse'):
   s=Settings();llm=MockClient() if mock else OllamaClient(s.base_url,s.model,s.max_image_side);res=PhishIntentionPipeline(llm,KnowledgeBase('knowledge'),s.threshold,s.top_k).run(tmp,mode);st.json(res.model_dump())
with tab2:
 path=Path('data/processed/manifest.csv')
 if not path.exists():st.info('Run scripts/prepare_data.py first.')
 else:
  df=pd.read_csv(path).fillna('');pending=df[df.annotation_status=='unlabelled'];st.write(f'{len(pending)} unlabelled of {len(df)}')
  if len(pending):
   idx=st.number_input('Pending row',0,len(pending)-1,0);row=pending.iloc[int(idx)];st.image(row.image_path,width=700);st.write({k:row[k] for k in ['sample_id','source','brand','phishing_status']})
   choices=st.multiselect('Intentions',INTENTS);annotator=st.text_input('Annotator');notes=st.text_area('Notes')
   if st.button('Save label'):
    mask=df.sample_id==row.sample_id
    for x in INTENTS:df.loc[mask,x]=1 if x in choices else 0
    df.loc[mask,'annotation_status']='labelled';df.loc[mask,'annotator']=annotator;df.loc[mask,'notes']=notes;df.to_csv(path,index=False);st.success('Saved. Refresh to continue.')
