import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phishintention.llm import MockClient
from phishintention.retrieval import KnowledgeBase
from phishintention.pipeline import PhishIntentionPipeline
def test_mock_pipeline():
 r=PhishIntentionPipeline(MockClient(),KnowledgeBase("knowledge")).run("tests/fixtures/sample_login.png");assert "credential_theft" in r.labels
