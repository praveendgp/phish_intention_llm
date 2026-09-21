"""Common scaffolding for every agent in the project.

Design rule enforced here
-------------------------
**Every agent that reasons about a screenshot is a vision-language agent and
receives the image itself.** Agents never work from a text transcript alone.
`VisionAgent.ask_json` therefore requires an `image_b64` argument, validates
that the configured model is multimodal, and records the fact on the AgentStep
so the UI can prove which agents actually looked at the page.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import Config, ModelSpec, load_config
from ..llm.ollama_client import (LLMResponse, OllamaClient, OllamaError,
                                 VisionCapabilityError)
from ..rag.retriever import KnowledgeRetriever
from ..schemas import AgentStep
from ..utils.json_parse import parse_json
from ..utils.logging import get_logger

log = get_logger(__name__)


CATEGORY_BLOCK = """The four phishing intention categories are:
1. Credential Theft - capture of authentication secrets (password, OTP, security answers).
2. Financial Fraud - capture of payment instruments or solicitation of a transfer/payment.
3. Malware Distribution - inducing download/installation/execution of a malicious payload.
4. Personal Information Harvesting - collection of identity data beyond credentials
   (name, DOB, address, national ID, phone, employment, health, ID documents).

Assign a category ONLY when you can see concrete evidence for it in the image.
A page may carry one, two or three intentions simultaneously; four is never
observed in practice."""

VISION_DIRECTIVE = """You are looking at an actual screenshot of a website.
Base every statement on what is visibly rendered in that image. Do not infer
content that is not shown, and do not rely on assumptions about the brand."""


class VisionAgent:
    """Base class: model selection, vision-enforced prompting, JSON recovery."""

    name: str = "Agent"
    layer: str = "Layer 0"
    stage: str = "framework"       # "manifest" | "framework"
    role: str = "vision"           # key under `models:` in config.yaml

    def __init__(
        self,
        config: Optional[Config] = None,
        client: Optional[OllamaClient] = None,
        retriever: Optional[KnowledgeRetriever] = None,
        model_role: Optional[str] = None,
    ):
        self.cfg = config or load_config()
        self.client = client or OllamaClient(self.cfg)
        self.retriever = retriever or KnowledgeRetriever(self.cfg)
        self.role = model_role or self.role
        self.spec: ModelSpec = self.cfg.model(self.role)
        self.require_vision = self.cfg.vision_for_all_agents

        if self.require_vision and not self.spec.vision:
            raise VisionCapabilityError(
                f"Agent '{self.name}' uses role '{self.role}' -> model "
                f"'{self.spec.model}', which is declared vision: false. "
                f"All agents in this project must analyse the screenshot with a "
                f"VLM. Fix config.yaml or set framework.vision_for_all_agents: false."
            )

    # ------------------------------------------------------------------
    @property
    def model_label(self) -> str:
        return self.spec.label or self.spec.model

    def new_step(self) -> AgentStep:
        return AgentStep(agent=self.name, layer=self.layer,
                         model=self.model_label, stage=self.stage,
                         vision=self.require_vision)

    # ------------------------------------------------------------------
    def ask_json(
        self,
        prompt: str,
        image_b64: str,
        system: Optional[str] = None,
        default: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Vision-grounded generation + tolerant JSON parse.

        `image_b64` is mandatory: this is what makes the agent a VLM agent.
        """
        if not image_b64:
            raise VisionCapabilityError(
                f"Agent '{self.name}' was invoked without a screenshot. Every "
                f"agent in this framework must analyse the image directly."
            )

        system_prompt = f"{system}\n\n{VISION_DIRECTIVE}" if system else VISION_DIRECTIVE
        response: LLMResponse = self.client.generate(
            self.spec, prompt=prompt, system=system_prompt,
            images=[image_b64], json_mode=True, temperature=temperature,
        )
        parsed = parse_json(response.text, default)
        parsed.setdefault("_meta", {})
        parsed["_meta"].update({
            "model": response.model,
            "latency": round(response.latency, 2),
            "had_image": response.had_image,
            "raw_response": response.text[:4000],
        })
        return parsed

    # ------------------------------------------------------------------
    def run_guarded(self, step: AgentStep, fn, *args, **kwargs):
        """Execute `fn`, recording timing and errors on `step`."""
        step.start()
        try:
            return fn(*args, **kwargs)
        except OllamaError as exc:
            log.error("%s failed: %s", self.name, exc)
            step.fail(str(exc))
            raise
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("%s crashed", self.name)
            step.fail(f"{type(exc).__name__}: {exc}")
            raise

    # ------------------------------------------------------------------
    @staticmethod
    def bullets(items: Any, limit: int = 8) -> str:
        values = items if isinstance(items, list) else []
        return "\n".join(f"- {v}" for v in values[:limit]) or "- (none recorded)"
