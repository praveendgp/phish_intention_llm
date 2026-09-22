"""Client for a local Ollama server, specialised for vision-language calls.

Only `/api/generate`, `/api/show` and `/api/tags` are used, so the official
`ollama` package is optional. Screenshots are passed as base64 PNG payloads in
the `images` field - this is the mechanism by which every agent in the framework
"sees" the page rather than reasoning over a text description of it.

Robustness
----------
Small open-weight VLMs fail in three characteristic ways. All are handled by
*changing* the request before retrying, because an identical retry can never
succeed:

1. **Token-repeat loop** (MiniCPM-V, Moondream)
       HTTP 500 {"error":"prediction aborted, token repeat limit reached"}
   -> capped `num_predict` and repeat penalties, then retry without the JSON
      grammar, then retry shorter and slightly warmer.

2. **Context overflow** (Granite-Vision 2B and other small-context models)
       HTTP 400 {"error":{"code":400,"message":"request (8268 tokens) exceeds
                 the available context size (8192 tokens)", ...}}
   -> raise `num_ctx` to fit (up to a ceiling), then down-scale the image
      (vision tokens dominate), then trim the prompt.

3. **Empty response** (small models over-constrained by the JSON grammar)
       200 OK with  {"response": ""}
   -> retry without the JSON grammar, then warmer, then with a shorter prompt
      and an explicit JSON primer. An empty generation is a *decoding* failure,
      not a transport failure, so plain retries are never used for it.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config, ModelSpec, load_config
from ..utils.logging import get_logger

log = get_logger(__name__)

_LOOP_MARKERS = ("token repeat limit reached", "prediction aborted", "repeat limit")
_CONTEXT_MARKERS = ("exceed_context_size", "exceeds the available context",
                    "context size", "n_ctx")

_TOKENS_RE = re.compile(r"request\s*\((\d+)\s*tokens?\)", re.I)
_NCTX_RE = re.compile(r'"n_ctx"\s*:\s*(\d+)')
_AVAIL_RE = re.compile(r"available context size\s*\((\d+)", re.I)

# Appended when a model returns nothing - nudges it to start emitting JSON.
_JSON_PRIMER = (
    "\n\nRespond now with the JSON object only. Begin your reply with the "
    "character { and end it with }. Do not add any commentary."
)


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot satisfy a request."""


class VisionCapabilityError(OllamaError):
    """Raised when a non-vision model is asked to analyse an image."""


class ContextOverflowError(OllamaError):
    """Raised when a prompt cannot be made to fit the model's context window."""


class EmptyResponseError(OllamaError):
    """Raised when a model keeps returning nothing at all."""


@dataclass
class LLMResponse:
    text: str
    model: str
    latency: float
    had_image: bool = False
    json_mode: bool = True
    degraded: bool = False          # JSON grammar had to be dropped
    shrunk_image: bool = False      # image was down-scaled to fit the context
    trimmed_prompt: bool = False    # prompt text was truncated
    primed: bool = False            # JSON primer was appended
    num_ctx: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Attempt:
    """One rung of a recovery ladder."""
    json_grammar: bool
    num_ctx: int
    num_predict: Optional[int] = None
    temp_bump: float = 0.0
    image_scale: float = 1.0
    prompt_budget: Optional[int] = None
    primer: bool = False
    reason: str = ""


class OllamaClient:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or load_config()
        self.host = self.cfg.ollama_host.rstrip("/")
        self.timeout = int(self.cfg.get("ollama.request_timeout", 300))
        self.keep_alive = self.cfg.get("ollama.keep_alive", "10m")
        self.num_ctx = int(self.cfg.get("ollama.num_ctx", 8192))
        self.max_retries = int(self.cfg.get("ollama.max_retries", 3))
        self.backoff = float(self.cfg.get("ollama.retry_backoff", 2.0))

        # --- anti-loop decoding defaults -------------------------------
        self.num_predict = int(self.cfg.get("ollama.num_predict", 1024))
        self.repeat_penalty = float(self.cfg.get("ollama.repeat_penalty", 1.15))
        self.repeat_last_n = int(self.cfg.get("ollama.repeat_last_n", 256))
        self.stop_sequences = list(self.cfg.get("ollama.stop", []) or [])
        self.no_json_grammar = {
            m.lower() for m in
            (self.cfg.get("ollama.disable_json_format_for", []) or [])
        }

        # --- context-overflow recovery ---------------------------------
        self.max_num_ctx = int(self.cfg.get("ollama.max_num_ctx", 16384))
        self.image_scale_ladder = [
            float(s) for s in
            (self.cfg.get("ollama.image_scale_ladder", [0.7, 0.5]) or [])
        ]
        self.chars_per_token = float(self.cfg.get("ollama.chars_per_token", 3.6))

        # --- empty-response recovery -----------------------------------
        # Temperature floor used when a model returns nothing: greedy decoding
        # at very low temperature is a common cause of empty generations.
        self.empty_temp_floor = float(
            self.cfg.get("ollama.empty_response_temperature", 0.35))

        # Telemetry
        self.vision_calls = 0
        self.text_calls = 0
        self.degraded_calls = 0
        self.context_recoveries = 0
        self.empty_recoveries = 0

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.host}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, endpoint: str) -> Dict[str, Any]:
        with urllib.request.urlopen(f"{self.host}{endpoint}", timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ------------------------------------------------------------------
    # health / inventory
    # ------------------------------------------------------------------
    def is_alive(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            data = self._get("/api/tags")
        except Exception as exc:  # pragma: no cover
            raise OllamaError(f"Cannot reach Ollama at {self.host}: {exc}") from exc
        return [m.get("name", "") for m in data.get("models", [])]

    def has_model(self, name: str) -> bool:
        available = self.list_models()
        if name in available:
            return True
        base = name.split(":")[0]
        return any(m.split(":")[0] == base for m in available)

    def supports_vision(self, name: str) -> Optional[bool]:
        try:
            data = self._post("/api/show", {"name": name})
        except Exception:
            return None
        families = (data.get("details", {}) or {}).get("families") or []
        caps = data.get("capabilities") or []
        blob = " ".join(str(x).lower() for x in list(families) + list(caps))
        if not blob:
            return None
        return any(k in blob for k in ("clip", "vision", "mllama", "vit", "image"))

    def context_length(self, name: str) -> Optional[int]:
        try:
            data = self._post("/api/show", {"name": name})
        except Exception:
            return None
        info = data.get("model_info") or {}
        for key, value in info.items():
            if key.endswith(".context_length"):
                try:
                    return int(value)
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # option assembly
    # ------------------------------------------------------------------
    def _role_opt(self, spec: ModelSpec, key: str, default: Any) -> Any:
        value = self.cfg.get(f"models.{spec.role}.{key}")
        return default if value is None else value

    def _wants_json_grammar(self, spec: ModelSpec, requested: bool) -> bool:
        if not requested:
            return False
        name = spec.model.lower()
        if name in self.no_json_grammar or name.split(":")[0] in self.no_json_grammar:
            return False
        return bool(self._role_opt(spec, "json_format", True))

    def _build_options(self, spec: ModelSpec, attempt: _Attempt,
                       temperature: Optional[float]) -> Dict[str, Any]:
        base_temp = temperature if temperature is not None else spec.temperature
        options: Dict[str, Any] = {
            "temperature": round(min(1.0, base_temp + attempt.temp_bump), 3),
            "top_p": spec.top_p,
            "num_ctx": attempt.num_ctx,
            "num_predict": int(attempt.num_predict
                               or self._role_opt(spec, "num_predict",
                                                 self.num_predict)),
            "repeat_penalty": float(self._role_opt(spec, "repeat_penalty",
                                                   self.repeat_penalty)),
            "repeat_last_n": int(self._role_opt(spec, "repeat_last_n",
                                                self.repeat_last_n)),
        }
        stop = self._role_opt(spec, "stop", self.stop_sequences)
        if stop:
            options["stop"] = list(stop)
        return options

    # ------------------------------------------------------------------
    # image handling
    # ------------------------------------------------------------------
    @staticmethod
    def _rescale_b64(image_b64: str, scale: float) -> str:
        """Down-scale a base64 PNG. Vision tokens usually dominate the prompt."""
        if scale >= 1.0:
            return image_b64
        try:
            from PIL import Image
        except Exception:  # pragma: no cover
            return image_b64
        try:
            raw = base64.b64decode(image_b64)
            with Image.open(io.BytesIO(raw)) as img:
                img = img.convert("RGB")
                width, height = img.size
                img = img.resize((max(64, int(width * scale)),
                                  max(64, int(height * scale))), Image.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                return base64.b64encode(buffer.getvalue()).decode("ascii")
        except Exception as exc:
            log.warning("Could not rescale image (%s); sending the original.", exc)
            return image_b64

    # ------------------------------------------------------------------
    # error inspection
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_context_error(detail: str) -> Tuple[Optional[int], Optional[int]]:
        required = available = None
        match = _TOKENS_RE.search(detail)
        if match:
            required = int(match.group(1))
        match = _NCTX_RE.search(detail) or _AVAIL_RE.search(detail)
        if match:
            available = int(match.group(1))
        return required, available

    @staticmethod
    def _is_context_error(detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in _CONTEXT_MARKERS)

    @staticmethod
    def _is_loop_error(detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in _LOOP_MARKERS)

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        json_mode: bool = True,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Single-turn generation. `images` are base64-encoded PNG strings."""
        if images and not spec.vision:
            raise VisionCapabilityError(
                f"Model '{spec.model}' (role '{spec.role}') is configured with "
                f"vision: false but was given a screenshot to analyse. Every "
                f"agent in this project must use a VLM - set vision: true and "
                f"point the role at a multimodal model."
            )

        role_ctx = int(self._role_opt(spec, "num_ctx", self.num_ctx))
        ceiling = int(self._role_opt(spec, "max_num_ctx", self.max_num_ctx))
        trained = self.context_length(spec.model)
        if trained:
            ceiling = min(ceiling, trained)
        ceiling = max(ceiling, role_ctx)

        queue: List[_Attempt] = [
            _Attempt(json_grammar=self._wants_json_grammar(spec, json_mode),
                     num_ctx=role_ctx, reason="initial")
        ]
        loop_used = context_used = empty_used = 0
        plain_retries = 0
        last_error: Optional[Exception] = None
        index = 0

        while index < len(queue):
            attempt = queue[index]
            index += 1

            payload_images = None
            if images:
                payload_images = [self._rescale_b64(img, attempt.image_scale)
                                  for img in images]

            send_prompt = prompt
            if attempt.prompt_budget and len(prompt) > attempt.prompt_budget:
                keep_head = int(attempt.prompt_budget * 0.65)
                keep_tail = attempt.prompt_budget - keep_head
                send_prompt = (prompt[:keep_head]
                               + "\n\n[... context trimmed to fit the model's "
                                 "context window ...]\n\n"
                               + prompt[-keep_tail:])
            if attempt.primer:
                send_prompt = send_prompt + _JSON_PRIMER

            payload: Dict[str, Any] = {
                "model": spec.model,
                "prompt": send_prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": self._build_options(spec, attempt, temperature),
            }
            if system:
                payload["system"] = system
            if payload_images:
                payload["images"] = payload_images
            if attempt.json_grammar:
                payload["format"] = "json"

            started = time.time()
            try:
                data = self._post("/api/generate", payload)
                text = (data.get("response") or "").strip()

                # ---------- empty generation ------------------------------
                if not text:
                    reason = data.get("done_reason") or "unknown"
                    step = self._next_empty_attempt(spec, attempt, empty_used,
                                                    temperature)
                    if step is not None:
                        empty_used += 1
                        log.warning(
                            "%s (role '%s') returned an empty response "
                            "(done_reason=%s) - %s.",
                            spec.model, spec.role, reason, step.reason)
                        queue.insert(index, step)
                        continue
                    raise EmptyResponseError(
                        f"Model '{spec.model}' (role '{spec.role}') returned an "
                        f"empty response on every attempt "
                        f"(done_reason={reason}).\n"
                        f"  This is a decoding failure, not a transport error.\n"
                        f"  Fixes:\n"
                        f"    (a) set  models.{spec.role}.json_format: false  "
                        f"- the JSON grammar over-constrains small models;\n"
                        f"    (b) raise  models.{spec.role}.temperature  to "
                        f"~0.3 - greedy decoding can emit only a stop token;\n"
                        f"    (c) lower  framework.max_image_edge  to 896 - an "
                        f"oversized image can crowd out the response budget;\n"
                        f"    (d) swap role '{spec.role}' for a sturdier VLM, "
                        f"e.g. gemma3:4b or gemma3:12b."
                    )

                if images:
                    self.vision_calls += 1
                else:
                    self.text_calls += 1
                degraded = (json_mode and not attempt.json_grammar)
                if degraded:
                    self.degraded_calls += 1
                if attempt.image_scale < 1.0 or attempt.prompt_budget:
                    self.context_recoveries += 1
                if attempt.primer or attempt.temp_bump:
                    self.empty_recoveries += 1

                return LLMResponse(
                    text=text, model=spec.model, latency=time.time() - started,
                    had_image=bool(images), json_mode=attempt.json_grammar,
                    degraded=degraded, shrunk_image=attempt.image_scale < 1.0,
                    trimmed_prompt=bool(attempt.prompt_budget),
                    primed=attempt.primer, num_ctx=attempt.num_ctx, raw=data,
                )

            except urllib.error.HTTPError as exc:  # pragma: no cover
                detail = exc.read().decode("utf-8", "ignore")[:600]
                last_error = OllamaError(f"HTTP {exc.code} from Ollama: {detail}")

                if exc.code == 404:
                    raise OllamaError(
                        f"Model '{spec.model}' is not available on the Ollama "
                        f"server. Pull it first:  ollama pull {spec.model}"
                    ) from exc

                # ---------- context overflow ---------------------------
                if self._is_context_error(detail):
                    required, current = self._parse_context_error(detail)
                    current = current or attempt.num_ctx
                    step = self._next_context_attempt(
                        attempt, required, current, ceiling, len(send_prompt))
                    if step is not None:
                        context_used += 1
                        log.warning(
                            "%s (role '%s'): prompt needs %s tokens but n_ctx is "
                            "%s - %s.", spec.model, spec.role,
                            required or "?", current, step.reason)
                        queue.insert(index, step)
                        continue
                    raise ContextOverflowError(
                        f"Prompt does not fit '{spec.model}' (role '{spec.role}').\n"
                        f"  Needed ~{required or '?'} tokens; ceiling is {ceiling}.\n"
                        f"  Fixes:\n"
                        f"    (a) raise  models.{spec.role}.num_ctx  (and "
                        f"ollama.max_num_ctx) if you have the VRAM;\n"
                        f"    (b) lower  framework.max_image_edge  (e.g. 896) - "
                        f"image tokens dominate this prompt;\n"
                        f"    (c) point role '{spec.role}' at a larger-context "
                        f"VLM, e.g. gemma3:12b.\n"
                        f"  Server said: {detail}"
                    ) from exc

                # ---------- degenerate loop ----------------------------
                if self._is_loop_error(detail):
                    step = self._next_loop_attempt(attempt, loop_used)
                    if step is not None:
                        loop_used += 1
                        log.warning("%s hit a token-repeat loop - %s.",
                                    spec.model, step.reason)
                        queue.insert(index, step)
                        continue
                    raise OllamaError(
                        f"Model '{spec.model}' (role '{spec.role}') keeps looping "
                        f"on this screenshot, even without the JSON grammar.\n"
                        f"  Try:  (a) set  models.{spec.role}.json_format: false;\n"
                        f"        (b) lower  ollama.num_predict;\n"
                        f"        (c) swap the model for this role.\n"
                        f"  Server said: {detail}"
                    ) from exc

            except (VisionCapabilityError, ContextOverflowError,
                    EmptyResponseError):
                # Actionable configuration failures - blind retries cannot help.
                raise
            except Exception as exc:  # pragma: no cover
                last_error = exc

            # ---------- generic retry (transport failures only) --------
            if index >= len(queue) and plain_retries < self.max_retries - 1:
                plain_retries += 1
                sleep_for = self.backoff * plain_retries
                log.warning("Ollama call failed (%s/%s) for %s: %s - retrying in "
                            "%.1fs", plain_retries + 1, self.max_retries,
                            spec.model, last_error, sleep_for)
                time.sleep(sleep_for)
                queue.append(_Attempt(json_grammar=attempt.json_grammar,
                                      num_ctx=attempt.num_ctx,
                                      num_predict=attempt.num_predict,
                                      image_scale=attempt.image_scale,
                                      prompt_budget=attempt.prompt_budget,
                                      primer=attempt.primer,
                                      reason="transient retry"))

        raise OllamaError(
            f"Generation failed for model '{spec.model}' (role '{spec.role}') "
            f"after {len(queue)} attempt(s): {last_error}"
        )

    # ------------------------------------------------------------------
    # recovery ladders
    # ------------------------------------------------------------------
    def _next_empty_attempt(self, spec: ModelSpec, attempt: _Attempt, used: int,
                            temperature: Optional[float]) -> Optional[_Attempt]:
        """Fixed three-rung ladder: drop the grammar, warm up, then shrink.

        The ladder is indexed by `used` (how many empty responses this call has
        already produced), so every rung is guaranteed to run exactly once even
        though the ladder is rebuilt on each failure.
        """
        base_temp = temperature if temperature is not None else spec.temperature
        warm = max(0.25, self.empty_temp_floor - base_temp)

        def rung(temp_bump: float, scale: float, budget: Optional[int],
                 predict: Optional[int], reason: str) -> _Attempt:
            return _Attempt(
                json_grammar=False,                 # never re-apply the grammar
                num_ctx=attempt.num_ctx,
                num_predict=predict if predict is not None else attempt.num_predict,
                temp_bump=temp_bump,
                image_scale=scale,
                prompt_budget=budget,
                primer=True,
                reason=reason,
            )

        smaller = min(0.7, attempt.image_scale * 0.7) if attempt.image_scale > 0.5 \
            else attempt.image_scale

        ladder = [
            # 1 - the JSON grammar over-constrains small models
            rung(0.0, attempt.image_scale, attempt.prompt_budget, None,
                 "retrying without the JSON grammar"),
            # 2 - greedy decoding can emit nothing but a stop token
            rung(warm, attempt.image_scale, attempt.prompt_budget, None,
                 f"retrying warmer (+{warm:.2f})"),
            # 3 - free up room for the model to actually speak
            rung(warm + 0.10, smaller,
                 min(attempt.prompt_budget or 10_000, 6000),
                 max(512, self.num_predict),
                 "retrying with a smaller image and a shorter prompt"),
        ]
        return ladder[used] if used < len(ladder) else None

    def _next_context_attempt(self, attempt: _Attempt, required: Optional[int],
                              current: int, ceiling: int,
                              prompt_chars: int) -> Optional[_Attempt]:
        """Grow the window, then shrink the image, then trim the prompt."""
        if required and current < ceiling:
            target = min(ceiling, max(current * 2, int(required * 1.15) + 256))
            target = int(round(target / 512.0) * 512)
            if target > current:
                return _Attempt(json_grammar=attempt.json_grammar, num_ctx=target,
                                num_predict=attempt.num_predict,
                                image_scale=attempt.image_scale,
                                prompt_budget=attempt.prompt_budget,
                                primer=attempt.primer,
                                reason=f"raising num_ctx to {target}")

        ladder = [s for s in self.image_scale_ladder if s < attempt.image_scale]
        if ladder:
            scale = ladder[0]
            return _Attempt(json_grammar=attempt.json_grammar,
                            num_ctx=min(ceiling, max(attempt.num_ctx, current)),
                            num_predict=attempt.num_predict, image_scale=scale,
                            prompt_budget=attempt.prompt_budget,
                            primer=attempt.primer,
                            reason=f"down-scaling the screenshot to {scale:.0%}")

        if attempt.prompt_budget is None and required:
            window = min(ceiling, max(current, attempt.num_ctx))
            overflow = max(256, required - window + 512)
            budget = max(1200, prompt_chars - int(overflow * self.chars_per_token))
            if budget < prompt_chars:
                return _Attempt(json_grammar=attempt.json_grammar, num_ctx=window,
                                num_predict=attempt.num_predict,
                                image_scale=attempt.image_scale,
                                prompt_budget=budget, primer=attempt.primer,
                                reason=f"trimming the prompt to ~{budget} chars")
        return None

    def _next_loop_attempt(self, attempt: _Attempt,
                           used: int) -> Optional[_Attempt]:
        ladder = [
            _Attempt(json_grammar=False, num_ctx=attempt.num_ctx,
                     num_predict=attempt.num_predict,
                     image_scale=attempt.image_scale,
                     prompt_budget=attempt.prompt_budget, primer=attempt.primer,
                     reason="retrying without the JSON grammar"),
            _Attempt(json_grammar=False, num_ctx=attempt.num_ctx,
                     num_predict=max(256, self.num_predict // 2), temp_bump=0.15,
                     image_scale=attempt.image_scale,
                     prompt_budget=attempt.prompt_budget, primer=attempt.primer,
                     reason="retrying shorter and slightly warmer"),
        ]
        return ladder[used] if used < len(ladder) else None
