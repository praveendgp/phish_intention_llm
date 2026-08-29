import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).parents[1]
        / "src"
    ),
)

from phishintention.llm import MockClient
from phishintention.pipeline import (
    PhishIntentionPipeline,
)
from phishintention.retrieval import KnowledgeBase


PROJECT_ROOT = Path(__file__).parents[1]


def create_pipeline():
    return PhishIntentionPipeline(
        llm=MockClient(),
        kb=KnowledgeBase(
            PROJECT_ROOT / "knowledge"
        ),
        threshold=0.55,
        top_k=3,
    )


def test_mock_pipeline():
    pipeline = create_pipeline()

    result = pipeline.run(
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "sample_login.png",
        mode="gated",
    )

    assert "credential_theft" in result.labels
    assert result.confidence == 0.87

    assert (
        "credential_theft"
        in result.evidence
    )

    assert (
        "vision_analysis"
        in result.agents_invoked
    )

    assert (
        "context_enrichment"
        in result.agents_invoked
    )

    assert (
        "classification"
        in result.agents_invoked
    )

    assert (
        "credential_theft_specialist"
        in result.agents_invoked
    )

    assert (
        "validator"
        in result.agents_invoked
    )

    assert (
        result.evidence_consistency
        == 1.0
    )


def test_single_agent_mock_pipeline():
    pipeline = create_pipeline()

    result = pipeline.run(
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "sample_login.png",
        mode="single",
    )

    assert result.labels == [
        "credential_theft"
    ]

    assert result.confidence == 0.80

    assert result.agents_invoked == [
        "single_agent"
    ]


def test_always_on_mock_pipeline():
    pipeline = create_pipeline()

    result = pipeline.run(
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "sample_login.png",
        mode="always",
    )

    assert "credential_theft" in result.labels

    expected_specialists = {
        "credential_theft_specialist",
        "financial_fraud_specialist",
        "malware_distribution_specialist",
        (
            "personal_information_harvesting"
            "_specialist"
        ),
    }

    assert expected_specialists.issubset(
        set(result.agents_invoked)
    )