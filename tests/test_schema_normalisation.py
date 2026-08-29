import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).parents[1]
        / "src"
    ),
)

from phishintention.schema import (
    AnalysisResult,
    normalise_confidence,
)


def test_numeric_confidence():
    assert normalise_confidence(0.81) == 0.81


def test_string_confidence():
    assert normalise_confidence("0.81") == 0.81


def test_percentage_confidence():
    assert normalise_confidence("81%") == 0.81


def test_single_item_list_confidence():
    assert normalise_confidence([0.81]) == 0.81


def test_dictionary_confidence():
    assert normalise_confidence(
        {
            "score": 0.81,
        }
    ) == 0.81


def test_list_of_confidence_objects():
    result = normalise_confidence(
        [
            {
                "intent": "credential_theft",
                "score": 0.81,
            },
            {
                "intent": "financial_fraud",
                "score": 0.52,
            },
        ]
    )

    assert result == 0.81


def test_analysis_result_accepts_list_confidence():
    result = AnalysisResult(
        sample_id="sample-1",
        labels=[
            "credential_theft",
        ],
        confidence=[
            0.88,
        ],
        evidence={
            "credential_theft": [
                "Password field is visible."
            ]
        },
        candidates=[],
        agents_invoked=[
            "validator",
        ],
        evidence_consistency=[
            0.90,
        ],
        trace={},
    )

    assert result.confidence == 0.88
    assert result.evidence_consistency == 0.90