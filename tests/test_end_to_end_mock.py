"""End-to-end test with a mocked Ollama server (no models required).

Verifies three things the project brief demands:

  1. EVERY agent - manifest stage and framework stage - receives the screenshot
     as an image payload (i.e. they are genuinely VLM agents).
  2. The annotator ensemble produces a manifest file.
  3. The framework's predictions are scored against that manifest.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishintentionllm.annotation import AnnotationStore, ManifestBuilder
from phishintentionllm.config import load_config
from phishintentionllm.evaluation import Evaluator
from phishintentionllm.llm.ollama_client import LLMResponse
from phishintentionllm.pipeline import BatchRunner, PhishIntentionLLM
from phishintentionllm.schemas import Sample


class MockClient:
    """Canned, role-aware JSON responses. Records whether each call had an image."""

    host = "mock://ollama"

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []            # (model, had_image, kind)
        self.vision_calls = 0
        self.text_calls = 0

    def is_alive(self):
        return True

    def list_models(self):
        return [m.model for m in self.cfg.all_models()]

    def has_model(self, name):
        return True

    def supports_vision(self, name):
        return True

    # ------------------------------------------------------------------
    def generate(self, spec, prompt, system=None, images=None, json_mode=True,
                 temperature=None):
        kind = self._kind(prompt)
        self.calls.append((spec.role, spec.model, bool(images), kind))
        if images:
            self.vision_calls += 1
        else:
            self.text_calls += 1
        payload = self._route(kind, spec, prompt)
        return LLMResponse(text=json.dumps(payload), model=spec.model,
                           latency=0.01, had_image=bool(images), raw={})

    # ------------------------------------------------------------------
    @staticmethod
    def _kind(prompt: str) -> str:
        if "report ONLY what is visually present" in prompt:
            return "vision"
        if "tag the suspicious elements" in prompt:
            return "context"
        if "score ALL FOUR categories" in prompt:
            return "classify"
        if "## Your domain:" in prompt:
            return "specialist"
        if "issue the final assessment" in prompt.lower():
            return "validate"
        if "labelling a website screenshot for a phishing" in prompt:
            return "annotate"
        if "rule on EACH disputed category" in prompt:
            return "tiebreak"
        if "issue the final manifest label set" in prompt:
            return "finalize"
        if "identify its phishing intention" in prompt:
            return "baseline"
        return "unknown"

    def _route(self, kind, spec, prompt):
        if kind == "vision":
            return {
                "ocr_text": "Sign in | Email | Password | Forgot password? | "
                            "Microsoft 365",
                "interface_elements": ["login card", "brand header"],
                "form_fields": ["Email", "Password"],
                "buttons": ["Sign in", "Next"],
                "branding": ["Microsoft"],
                "layout": "Single centred login card on a plain background.",
                "domain_hints": ["ms-verify-login.top"],
                "has_password_field": True, "has_payment_field": False,
                "has_download_action": False, "has_identity_fields": False,
                "sector": "online/cloud service", "image_quality": "clear",
            }
        if kind == "context":
            return {
                "tagged_elements": [{"element": "Password field",
                                     "tag": "credential capture",
                                     "implication": "account takeover"}],
                "missed_elements": ["fake padlock icon"],
                "security_implications": ["Password entry on a non-Microsoft domain"],
                "hypotheses": ["Credential Theft via fake Microsoft 365 login"],
                "deception_signals": ["brand-locked sign-in card"],
                "phishing_likelihood": 0.95, "notes": "",
            }
        if kind == "classify":
            return {
                "scores": {"Credential Theft": 0.93, "Financial Fraud": 0.05,
                           "Malware Distribution": 0.02,
                           "Personal Information Harvesting": 0.22},
                "evidence": {"Credential Theft": ["Password field", "Microsoft logo"]},
                "nominated": ["Credential Theft"],
                "sector": "online/cloud service",
                "reasoning": "Only a credential form is present.",
            }
        if kind == "specialist":
            confirmed = "Credential Theft" in prompt.split("## Your domain:")[1][:60]
            return {"confirmed": confirmed,
                    "confidence": 0.93 if confirmed else 0.12,
                    "evidence": ["Password field"] if confirmed else [],
                    "matched_indicators": ["visible password input"] if confirmed else [],
                    "reasoning": "Password capture is the page's sole function."}
        if kind == "validate":
            return {
                "final_intentions": [{
                    "category": "Credential Theft", "confidence": 0.93,
                    "evidence": ["Password field", "Microsoft logo",
                                 "domain ms-verify-login.top"],
                    "explanation": "The page exists to capture your Microsoft "
                                   "email and password."}],
                "is_phishing": True, "phishing_score": 0.96,
                "verdict": "Fake Microsoft 365 login harvesting credentials",
                "sector": "online/cloud service", "overall_confidence": 0.93,
                "risk_summary": "The attacker wants Microsoft 365 credentials.",
                "recommended_action": "Block the domain and reset exposed passwords.",
                "conflicts_resolved": [],
            }
        if kind == "annotate":
            intentions = [{"category": "Credential Theft", "confidence": 0.92,
                           "evidence": ["Password field"]}]
            # Annotator B additionally claims PIH -> forces a tie-break.
            if spec.role == "annotator_b":
                intentions.append({"category": "Personal Information Harvesting",
                                   "confidence": 0.55, "evidence": ["Email field"]})
            return {"intentions": intentions, "sector": "online/cloud service",
                    "is_phishing": True, "phishing_confidence": 0.94,
                    "rationale": "Brand-locked login form on an unrelated domain."}
        if kind == "tiebreak":
            return {"decisions": {"Personal Information Harvesting": False},
                    "confidences": {"Personal Information Harvesting": 0.25},
                    "evidence": {},
                    "reasoning": "Only an email identifier is requested."}
        if kind == "finalize":
            return {
                "labels": [{"category": "Credential Theft", "confidence": 0.94,
                            "evidence": ["Password field on ms-verify-login.top"],
                            "source": "both"}],
                "rejected": [{"category": "Personal Information Harvesting",
                              "reason": "no identity fields visible"}],
                "sector": "online/cloud service", "is_phishing": True,
                "label_quality": "high", "needs_human_review": False,
                "summary": "Fake Microsoft 365 sign-in harvesting credentials.",
            }
        if kind == "baseline":
            return {"intentions": [{"category": "Credential Theft",
                                    "confidence": 0.8, "evidence": ["login form"]}],
                    "sector": "online/cloud service", "is_phishing": True,
                    "phishing_score": 0.8, "verdict": "Fake login page"}
        return {}


def make_test_image(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (900, 600), "#f3f4f6")
        draw = ImageDraw.Draw(img)
        draw.rectangle([250, 150, 650, 450], fill="white", outline="#d1d5db")
        draw.text((300, 190), "Microsoft", fill="#0078d4")
        draw.text((300, 250), "Email", fill="#374151")
        draw.rectangle([300, 270, 600, 300], outline="#9ca3af")
        draw.text((300, 320), "Password", fill="#374151")
        draw.rectangle([300, 340, 600, 370], outline="#9ca3af")
        draw.rectangle([300, 395, 400, 425], fill="#0078d4")
        draw.text((320, 405), "Sign in", fill="white")
        img.save(path)
    except Exception:
        path.write_bytes(b"\x89PNG\r\n\x1a\n")


def main() -> int:
    print("\n" + "=" * 72)
    print("  PhishIntentionLLM - end-to-end test (mocked VLM server)")
    print("=" * 72)

    cfg = load_config()
    tmp = Path(tempfile.mkdtemp())
    image = tmp / "login.png"
    make_test_image(image)

    store = AnnotationStore(cfg)
    store.manifest_path = tmp / "manifest.jsonl"
    store.manifest_runs_dir = tmp / "manifest_runs"
    store.predictions_path = tmp / "predictions.jsonl"
    store.runs_dir = tmp / "runs"
    store.gt_path = tmp / "gt.jsonl"

    samples = [Sample(sample_id=f"putra::phishing::demo{i}", source="putra",
                      screenshot_path=str(image)) for i in range(3)]

    # ------------------------------------------------------------------
    # STAGE A - annotators build the manifest
    # ------------------------------------------------------------------
    print("\n[STAGE A] Annotator ensemble -> manifest")
    client = MockClient(cfg)
    builder = ManifestBuilder(cfg, client=client, store=store)
    manifest_payload = builder.build(samples)

    record = builder.annotate_sample(samples[0])
    print("  Agent flow (manifest stage):")
    for step in record.steps:
        flag = "IMG" if step.vision else "txt"
        print(f"    [{step.status:<7}] [{flag}] {step.layer:<34} {step.agent:<22} "
              f"{step.summary[:44]}")

    assert manifest_payload["n_success"] == 3
    assert record.labels == ["Credential Theft"], record.labels
    assert record.tie_break_used, "tie-breaker should have fired"
    assert record.agreement == "partial agreement"
    assert len(record.votes) == 2
    assert store.manifest_path.exists(), "manifest file must be written"
    print(f"\n  Manifest written  : {store.manifest_path.name} "
          f"({len(store.manifest_ids())} records)")
    print(f"  Reference labels  : {record.labels}")
    print(f"  Agreement         : {record.agreement} | tie-break: "
          f"{record.tie_break_used}")
    print(f"  Annotator models  : {manifest_payload['annotator_models']}")

    # ------------------------------------------------------------------
    # STAGE B - framework predicts (never sees the manifest)
    # ------------------------------------------------------------------
    print("\n[STAGE B] Five-layer framework -> predictions")
    framework = PhishIntentionLLM(cfg, client=client)
    result = framework.analyse_sample(samples[0])

    print("  Agent flow (framework stage):")
    for step in result.steps:
        flag = "IMG" if step.vision else "txt"
        print(f"    [{step.status:<7}] [{flag}] {step.layer:<34} {step.agent:<34} "
              f"{step.summary[:40]}")

    assert not result.error, result.error
    assert result.labels == ["Credential Theft"], result.labels
    assert result.verdict.startswith("Fake Microsoft")
    assert result.vision_calls >= 5, f"only {result.vision_calls} vision calls"
    print(f"\n  Verdict           : {result.verdict}")
    print(f"  Vision calls made : {result.vision_calls}")

    runner = BatchRunner(cfg, framework, store)
    batch = runner.run(samples, mode="framework")
    assert batch["n_success"] == 3
    assert batch["n_scored_against_manifest"] == 3

    # ------------------------------------------------------------------
    # STAGE C - every agent must have received the image
    # ------------------------------------------------------------------
    print("\n[STAGE C] VLM enforcement audit")
    seen: dict = {}
    for role, model, had_image, kind in client.calls:
        entry = seen.setdefault(role, {"model": model, "calls": 0, "with_image": 0})
        entry["calls"] += 1
        entry["with_image"] += int(had_image)

    for role, info in sorted(seen.items()):
        mark = "OK " if info["with_image"] == info["calls"] else "FAIL"
        print(f"    [{mark}] {role:<20} {info['model']:<26} "
              f"{info['with_image']}/{info['calls']} calls carried the image")
        assert info["with_image"] == info["calls"], \
            f"agent role '{role}' made a text-only call"

    assert client.text_calls == 0, "some agent reasoned without the screenshot"
    expected_roles = {"annotator_a", "annotator_b", "manifest_tiebreaker",
                      "manifest_finalizer", "vision", "context", "classifier",
                      "specialist", "validator"}
    assert expected_roles.issubset(set(seen)), \
        f"roles never invoked: {expected_roles - set(seen)}"
    print(f"\n  Total VLM calls   : {client.vision_calls} "
          f"(text-only calls: {client.text_calls})")

    # ------------------------------------------------------------------
    # STAGE D - evaluation against the manifest
    # ------------------------------------------------------------------
    print("\n[STAGE D] Evaluation: predictions scored against the manifest")
    evaluator = Evaluator(cfg, store)
    outcome = evaluator.evaluate_run()
    assert "error" not in outcome, outcome
    report = outcome["report"]
    print(f"    reference        : {outcome['reference']}")
    print(f"    samples scored   : {outcome['n_evaluated']}")
    print(f"    precision (micro): {report['precision_micro']:.4f}")
    print(f"    recall    (micro): {report['recall_micro']:.4f}")
    print(f"    F1        (micro): {report['f1_micro']:.4f}")
    print(f"    overall accuracy : {report['overall_accuracy']:.4f}")
    print(f"    agreement        : {outcome['agreement']}")

    assert outcome["reference"] == "manifest"
    assert outcome["n_evaluated"] == 3
    assert report["precision_micro"] == 1.0

    quality = evaluator.manifest_quality()
    print(f"\n  Manifest quality  : IAA={quality['inter_annotator_agreement']}, "
          f"tie-breaks={quality['tie_breaks']}, review={quality['needs_review']}")

    rows = evaluator.per_sample()
    assert rows and rows[0]["outcome"] == "exact"

    print("\n" + "=" * 72)
    print("  All end-to-end assertions passed.")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
