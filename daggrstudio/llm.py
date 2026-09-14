"""
The LLM side of Daggr Studio: one small client, BYOK-first, with a hard community-pool cap.

Cost rules (see MASTER_PLAN §3):
* the user's own HF token is used whenever one is supplied (BYOK);
* without a token we fall back to the Space's own token, but only until the daily
  community-pool counter is exhausted - enforced here, not in the UI;
* provider routing goes through HF Inference Providers, so a user can also plug in
  OpenAI/Anthropic/Groq/OpenRouter keys via the same interface.

Model choice was verified live on the HF router (2026-09-14): DeepSeek-V4.1-Flash is
~1s and emits clean JSON, which is what the planner and the medic need.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.environ.get("DAGGRSTUDIO_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")

#: Tried in order when the chosen model errors out (verified reachable 2026-09-14).
FALLBACK_CHAIN = [
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "zai-org/GLM-4.7-Flash",
    "Qwen/Qwen3-4B-Instruct-2507",
    "deepseek-ai/DeepSeek-V3.2",
    "meta-llama/Llama-3.1-8B-Instruct",
]

#: Curated shortlist for the model dropdown (all router-available).
SUGGESTED_MODELS = [
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-4.7-Flash",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

POOL_CAP = int(os.environ.get("DAGGRSTUDIO_POOL_DAILY_CAP", 40))
POOL_PATH = Path(
    os.environ.get("DAGGRSTUDIO_CACHE", Path.home() / ".cache" / "daggr-studio" / "pool.json")
)

MODELS_CACHE_TTL = 3600
_MODELS_CACHE: tuple[float, list[str]] | None = None


class PoolExhausted(RuntimeError):
    """Raised when no BYOK token is present and the shared pool is spent for today."""


class LLMError(RuntimeError):
    """All models in the chain failed."""


@dataclass
class CallStats:
    calls: int = 0
    failures: int = 0
    seconds: float = 0.0
    models_used: list[str] = field(default_factory=list)
    last_error: str = ""

    def note(self, model: str, seconds: float) -> None:
        self.calls += 1
        self.seconds += seconds
        if model and (not self.models_used or self.models_used[-1] != model):
            self.models_used.append(model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "seconds": round(self.seconds, 2),
            "models_used": self.models_used,
            "last_error": self.last_error,
        }


# ─── community pool (the only place the platform spends money) ────────────────


def _read_pool() -> dict[str, Any]:
    try:
        return json.loads(POOL_PATH.read_text())
    except Exception:
        return {}


def pool_status() -> dict[str, Any]:
    data = _read_pool()
    today = date.today().isoformat()
    used = int(data.get(today, 0)) if data.get("date") == today else 0
    return {"cap": POOL_CAP, "used": used, "remaining": max(0, POOL_CAP - used), "date": today}


def pool_spend(n: int = 1) -> None:
    data = _read_pool()
    today = date.today().isoformat()
    used = int(data.get(today, 0)) if data.get("date") == today else 0
    if used + n > POOL_CAP:
        raise PoolExhausted(
            f"Community pool used up for today ({used}/{POOL_CAP}). "
            "Add your own Hugging Face token in Settings to keep going."
        )
    try:
        POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        POOL_PATH.write_text(json.dumps({"date": today, today: used + n}))
    except Exception:
        pass


# ─── token discovery ──────────────────────────────────────────────────────────


def resolve_token(user_token: str | None) -> tuple[str | None, bool]:
    """
    Return (token, is_byok). The user's token always wins; otherwise fall back to the
    Space's own token (pool-metered).
    """
    if user_token and user_token.strip():
        return user_token.strip(), True
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val, False
    try:
        cached = Path.home() / ".cache" / "huggingface" / "token"
        if cached.exists():
            return cached.read_text().strip(), False
    except Exception:
        pass
    return None, False


def verify_token(token: str) -> str | None:
    """Return the HF username for a token, or None when it is invalid."""
    if not token:
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi(token=token.strip()).whoami().get("name")
    except Exception:
        return None


# ── client ───────────────────────────────────────────────────────────────────


class LLMClient:
    """Small chat client with a fallback chain and pool accounting."""

    def __init__(
        self,
        token: str | None = None,
        provider: str = "auto",
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        self.raw_token = token
        self.provider = provider
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stats = CallStats()

    # ── plumbing ───────────────────────────────────────────────────────────────

    def _client(self, model: str, token: str):
        from huggingface_hub import InferenceClient

        return InferenceClient(model=model, provider=self.provider, token=token)

    def _chain(self) -> list[str]:
        chain = [self.model]
        for model in FALLBACK_CHAIN:
            if model not in chain:
                chain.append(model)
        return chain

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, str]:
        """Return (text, model_that_answered). Tries the fallback chain on failure."""
        token, is_byok = resolve_token(self.raw_token)
        if not token:
            raise LLMError(
                "No Hugging Face token available. Paste one in Settings (it is used only for "
                "your own requests and never stored)."
            )
        if not is_byok:
            pool_spend(1)  # raises PoolExhausted when the shared budget is gone

        first = model or self.model
        chain = [first] + [m for m in self._chain() if m != first]
        last_error = ""
        for candidate in chain:
            started = time.time()
            try:
                client = self._client(candidate, token)
                result = client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                )
                text = (result.choices[0].message.content or "").strip()
                if not text:
                    raise RuntimeError("empty completion")
                self.stats.note(candidate, time.time() - started)
                return text, candidate
            except Exception as exc:
                last_error = f"{candidate}: {type(exc).__name__}: {str(exc)[:200]}"
                self.stats.failures += 1
                self.stats.last_error = last_error
                continue
        raise LLMError(f"every model in the fallback chain failed. Last: {last_error}")

    def ask(self, prompt: str, system: str | None = None, **kwargs: Any) -> tuple[str, str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def ask_json(self, prompt: str, system: str | None = None, **kwargs: Any) -> tuple[Any, str]:
        """Ask for JSON and parse tolerantly; returns (parsed_or_None, model)."""
        from daggrstudio.jsonutil import parse_json

        text, model = self.ask(prompt, system=system, **kwargs)
        return parse_json(text), model


def list_router_models(token: str | None = None, limit: int = 400) -> list[str]:
    """Live list of models on the HF router (cached for an hour). Best-effort."""
    global _MODELS_CACHE
    if _MODELS_CACHE and time.time() - _MODELS_CACHE[0] < MODELS_CACHE_TTL:
        return _MODELS_CACHE[1]
    resolved, _ = resolve_token(token)
    if not resolved:
        return list(SUGGESTED_MODELS)
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://router.huggingface.co/v1/models",
            headers={"Authorization": f"Bearer {resolved}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read())
        models = sorted(m["id"] for m in payload.get("data", []) if m.get("id"))
        _MODELS_CACHE = (time.time(), models[:limit])
        return _MODELS_CACHE[1]
    except Exception:
        return list(SUGGESTED_MODELS)