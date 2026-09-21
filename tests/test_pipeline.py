"""Offline unit tests - no Ollama server required."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishintentionllm.annotation import (AnnotationStore, ManualAnnotationManager,
                                          analyse_agreement)
from phishintentionllm.config import ALL_ROLES, FRAMEWORK_ROLES, MANIFEST_ROLES, load_config
from phishintentionllm.datasets import DatasetRegistry
from phishintentionllm.evaluation import (agreement_rate, co_occurrence, evaluate,
                                          evaluate_binary)
from phishintentionllm.llm.ollama_client import VisionCapabilityError
from phishintentionllm.rag.knowledge_base import get_knowledge_base
from phishintentionllm.rag.retriever import KnowledgeRetriever
from phishintentionllm.schemas import AnnotatorVote, Intention, ManifestRecord
from phishintentionllm.utils.json_parse import as_bool, as_float, as_list, parse_json


def test_intention_coercion():
    assert Intention.coerce("credentials theft") == Intention.CREDENTIAL_THEFT
    assert Intention.coerce("CT") == Intention.CREDENTIAL_THEFT
    assert Intention.coerce("personal_information_harvesting") == Intention.PERSONAL_INFO
    assert Intention.coerce("Malware-Distribution") == Intention.MALWARE_DISTRIBUTION
    assert Intention.coerce("totally unrelated text") is None
    assert Intention.short("Financial Fraud") == "FF"
    print("  intention coercion .............. ok")


def test_json_parsing():
    assert parse_json('```json\n{"a": 1,}\n```')["a"] == 1
    assert parse_json('blah {"x": [1,2]} tail')["x"] == [1, 2]
    assert parse_json("not json at all", {"d": 1}) == {"d": 1}
    assert as_float("85%") == 0.85 and as_float(8.5) == 0.85 and as_float(0.42) == 0.42
    assert as_list("a, b; c") == ["a", "b", "c"]
    assert as_bool("yes") and not as_bool("no")
    print("  tolerant json parsing ........... ok")


def test_every_model_is_a_vlm():
    """Project constraint: all agents must analyse the image with a VLM."""
    cfg = load_config()
    for role in ALL_ROLES:
        spec = cfg.model(role)
        assert spec.vision, f"role '{role}' -> '{spec.model}' is not vision-enabled"
    banned = ("llama", "gpt-4o", "gemini", "qwen2.5-vl")
    for spec in cfg.all_models():
        name = spec.model.lower()
        assert not any(b in name for b in banned), f"disallowed model: {spec.model}"
    assert set(MANIFEST_ROLES).isdisjoint(FRAMEWORK_ROLES)
    print("  all roles are VLMs + constraints  ok")


def test_vision_is_enforced_at_the_agent_boundary():
    from phishintentionllm.agents.vision_agent import VisionAnalysisAgent

    class _Dummy:
        host = "mock://x"

        def generate(self, *a, **k):
            raise AssertionError("should never be reached")

    agent = VisionAnalysisAgent(config=load_config(), client=_Dummy())
    try:
        agent.ask_json("prompt", "")          # empty image must be rejected
        raise AssertionError("expected VisionCapabilityError")
    except VisionCapabilityError:
        pass
    print("  image-required guard ............ ok")


def test_knowledge_and_retrieval():
    kb = get_knowledge_base()
    assert kb.stats()["categories"] == 4
    assert kb.retrieve_specialist_knowledge("Credential Theft")["I_c_indicators"]
    retriever = KnowledgeRetriever(kb=kb)
    hits = retriever.retrieve("password field login form", top_k=3)
    assert hits and any("password" in h.text.lower() for h in hits)
    scoped = retriever.retrieve("download exe", top_k=3,
                                category="Malware Distribution")
    assert all(p.category in (None, "Malware Distribution") for p in scoped)
    print("  knowledge base + retrieval ...... ok")


def test_annotator_agreement_logic():
    a = AnnotatorVote("Annotator A", "m1", ["Credential Theft"],
                      {"Credential Theft": 0.9})
    b = AnnotatorVote("Annotator B", "m2", ["Credential Theft"],
                      {"Credential Theft": 0.8})
    full = analyse_agreement([a, b])
    assert full["status"] == "full agreement" and not full["requires_tiebreak"]

    c = AnnotatorVote("Annotator B", "m2",
                      ["Credential Theft", "Financial Fraud"],
                      {"Credential Theft": 0.8, "Financial Fraud": 0.7})
    partial = analyse_agreement([a, c])
    assert partial["status"] == "partial agreement"
    assert partial["disputed"] == ["Financial Fraud"]
    assert partial["requires_tiebreak"]

    d = AnnotatorVote("Annotator B", "m2", ["Malware Distribution"],
                      {"Malware Distribution": 0.8})
    conflict = analyse_agreement([a, d])
    assert conflict["status"] == "conflict" and conflict["requires_tiebreak"]

    # A decisively confident solo claim does not need a tie-break.
    e = AnnotatorVote("Annotator B", "m2",
                      ["Credential Theft", "Financial Fraud"],
                      {"Credential Theft": 0.8, "Financial Fraud": 0.97})
    assert not analyse_agreement([a, e])["requires_tiebreak"]
    print("  annotator consensus logic ....... ok")


def test_manifest_record_roundtrip():
    record = ManifestRecord(
        sample_id="putra::phishing::x", source="putra",
        screenshot_path="/tmp/x.png", labels=["Credential Theft"],
        confidences={"Credential Theft": 0.9}, agreement="full agreement")
    record.votes = [AnnotatorVote("Annotator A", "m1", ["Credential Theft"],
                                  {"Credential Theft": 0.9})]
    restored = ManifestRecord.from_dict(record.to_dict())
    assert restored.sample_id == record.sample_id
    assert restored.labels == ["Credential Theft"]
    assert restored.n_intentions == 1
    assert restored.votes[0].annotator == "Annotator A"
    print("  manifest record serialisation ... ok")


def test_metrics():
    y_true = [["Credential Theft"],
              ["Credential Theft", "Personal Information Harvesting"],
              ["Financial Fraud"],
              ["Credential Theft", "Financial Fraud",
               "Personal Information Harvesting"]]
    y_pred = [["Credential Theft"],
              ["Credential Theft"],
              ["Financial Fraud", "Malware Distribution"],
              ["Credential Theft", "Financial Fraud"]]
    report = evaluate(y_true, y_pred).to_dict()
    assert report["n_samples"] == 4
    assert report["overall_accuracy"] == 0.25          # 1 of 4 exact matches
    assert report["acc_by_complexity"]["2 intention(s)"] == 1.0   # t_k = 1
    assert report["acc_by_complexity"]["3 intention(s)"] == 1.0   # t_k = 2
    binary = evaluate_binary(y_true, y_pred)
    assert binary["tp"] == 3 and binary["fn"] == 0
    assert co_occurrence(y_true)["CT + PIH"] >= 1
    rates = agreement_rate(y_true, y_pred)
    assert rates["exact_match"] == 0.25 and rates["no_overlap"] == 0.0
    print("  evaluation metrics .............. ok")


def test_manifest_store_and_reference():
    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        store = AnnotationStore(cfg)
        store.manifest_path = Path(tmp) / "manifest.jsonl"
        store.predictions_path = Path(tmp) / "pred.jsonl"
        store.gt_path = Path(tmp) / "gt.jsonl"
        store.manual_path = Path(tmp) / "manual.jsonl"

        store.append_manifest(ManifestRecord(
            sample_id="s1", source="putra", screenshot_path="/tmp/a.png",
            labels=["Credential Theft"]))
        # A later line for the same sample supersedes the earlier one.
        store.append_manifest(ManifestRecord(
            sample_id="s1", source="putra", screenshot_path="/tmp/a.png",
            labels=["Credential Theft", "Financial Fraud"]))
        store.append_manifest(ManifestRecord(
            sample_id="s2", source="phishiris", screenshot_path="/tmp/b.png",
            labels=["Malware Distribution"]))

        mapping = store.manifest_map()
        assert len(mapping) == 2
        assert mapping["s1"] == ["Credential Theft", "Financial Fraud"]
        assert store.reference_map() == mapping           # default reference

        csv_path = store.export_manifest_csv(Path(tmp) / "manifest.csv")
        assert csv_path.exists() and "Credential Theft" in csv_path.read_text()
    print("  manifest store + reference ...... ok")


def test_manual_annotation_and_manifest_verification():
    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        store = AnnotationStore(cfg)
        store.manual_path = Path(tmp) / "manual.jsonl"
        store.gt_path = Path(tmp) / "gt.jsonl"
        store.manifest_path = Path(tmp) / "manifest.jsonl"
        manager = ManualAnnotationManager(cfg, store)

        sid = "putra::phishing::demo"
        manager.annotate(sid, "engineer_1", ["Credential Theft"], "financial")
        assert manager.status(sid)["state"] == "partially labelled"

        manager.annotate(sid, "engineer_2",
                         ["Credential Theft", "Personal Information Harvesting"],
                         "financial")
        status = manager.status(sid)
        assert status["state"] == "awaiting review"
        assert status["annotator_agreement"] is False

        manager.annotate(sid, "reviewer", ["Credential Theft"], "financial",
                         "password only")
        assert manager.status(sid)["state"] == "reviewed"
        labels, basis = manager.consensus(sid)
        assert labels == ["Credential Theft"] and basis == "reviewer"

        rows = manager.build_ground_truth()
        assert rows and rows[0]["labels"] == ["Credential Theft"]
        assert store.write_ground_truth(rows) == 1

        # Human verification of an annotator-generated manifest row.
        store.append_manifest(ManifestRecord(
            sample_id=sid, source="putra", screenshot_path="/tmp/a.png",
            labels=["Credential Theft", "Financial Fraud"], needs_review=True))
        assert manager.verify_manifest_row(sid, ["Credential Theft"],
                                           notes="no payment fields")
        row = store.manifest_record(sid)
        assert row["labels"] == ["Credential Theft"]
        assert row["human_verified"] and not row["needs_review"]
    print("  manual annotation + verification  ok")


def test_dataset_registry_handles_missing_sources():
    registry = DatasetRegistry(load_config())
    assert {d["source"] for d in registry.available()} == {"putra", "phishiris"}
    assert isinstance(registry.load_all(), list)
    print("  dataset registry ................ ok")


if __name__ == "__main__":
    print("\nPhishIntentionLLM - offline test suite\n")
    test_intention_coercion()
    test_json_parsing()
    test_every_model_is_a_vlm()
    test_vision_is_enforced_at_the_agent_boundary()
    test_knowledge_and_retrieval()
    test_annotator_agreement_logic()
    test_manifest_record_roundtrip()
    test_metrics()
    test_manifest_store_and_reference()
    test_manual_annotation_and_manifest_verification()
    test_dataset_registry_handles_missing_sources()
    print("\nAll tests passed.\n")
