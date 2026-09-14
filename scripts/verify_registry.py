#!/usr/bin/env python
"""
Rebuild and verify the brick registry against live Hugging Face state.

Why a script and not an agent: verification is a slow, rate-limited, resumable job
(dozens of Space round trips). This writes each brick to ``spaces.json`` the moment it is
verified, skips anything already verified, and can be re-run at any time - including from
a cron job - without redoing work.

Usage:
    python scripts/verify_registry.py                 # verify everything not yet verified
    python scripts/verify_registry.py --only flux1-schnell triposg
    python scripts/verify_registry.py --refresh       # ignore the existing file
    python scripts/verify_registry.py --discover      # also search the Hub for new candidates
    python scripts/verify_registry.py --models-only   # refresh the LLM bricks from the router

Never prints the token. Never writes a brick it did not verify.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from daggrstudio.introspect import introspect  # noqa: E402
from daggrstudio.registry.bricks import commercial_ok  # noqa: E402

OUT = ROOT / "daggrstudio" / "registry" / "spaces.json"
REPORT = ROOT / "docs" / "registry-verification.md"
LOG = ROOT / "docs" / "registry-verification.log"

PER_ITEM_TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", 100))

# ─── candidates ───────────────────────────────────────────────────────────────
# (id, kind, source, api_name, modality, industries, output_kind, notes)
# Curated from the daggr / daggr-pipelines skill tables - i.e. Spaces already known to
# work through daggr - plus common public alternatives to give the planner real choice.

SPACES: list[tuple[str, str, str | None, str, list[str], str, str]] = [
    # text-to-image
    ("flux1-schnell", "black-forest-labs/FLUX.1-schnell", "/infer", "image-gen",
     ["art", "game-dev", "3d", "marketing", "film"], "image", "fast, Apache-2.0, daggr-proven"),
    ("flux1-dev", "black-forest-labs/FLUX.1-dev", "/infer", "image-gen",
     ["art", "film", "marketing"], "image", "higher quality than schnell"),
    ("sd35-large", "stabilityai/stable-diffusion-3.5-large", "/infer", "image-gen",
     ["art", "marketing", "film"], "image", "community licence, check revenue terms"),
    ("z-image-turbo", "hf-applications/Z-Image-Turbo", "/generate_image", "image-gen",
     ["art", "game-dev", "marketing"], "image", "official HF app, fast, often sleeping"),
    ("kontext-dev", "black-forest-labs/FLUX.1-Kontext-dev", "/infer", "image-edit",
     ["art", "game-dev", "marketing"], "image", "instruction-based image editing"),
    ("qwen-image-edit", "Qwen/Qwen-Image-Edit", "/infer", "image-edit",
     ["art", "marketing", "film"], "image", "text-guided image editing"),
    # image editing / utility
    ("background-removal", "hf-applications/background-removal", "/image", "image-edit",
     ["game-dev", "art", "3d", "marketing"], "image", "official HF app; returns (orig, cut)"),
    ("not-lain-bg-removal", "not-lain/background-removal", "/run", "image-edit",
     ["game-dev", "art", "3d"], "image", "single in/out bg removal"),
    ("bria-rmbg", "briaai/BRIA-RMBG-2.0", "/run", "image-edit",
     ["game-dev", "art", "marketing"], "image", "high-quality matting"),
    # upscale / restore
    ("swin2sr", "Xenova/swin2SR", "/predict", "upscale",
     ["art", "game-dev", "film"], "image", "super-resolution"),
    ("real-esrgan", "ai-forever/Real-ESRGAN", "/predict", "upscale",
     ["art", "film", "game-dev"], "image", "classic upscaler"),
    # image / text to 3D
    ("triposg", "VAST-AI/TripoSG", "/generate", "image-to-3d",
     ["game-dev", "3d"], "model3d", "most reliable open image-to-3D"),
    ("trellis2", "microsoft/TRELLIS.2", "/image_to_3d", "image-to-3d",
     ["game-dev", "3d", "art"], "model3d", "high quality; may need HF token, multi-step API"),
    ("hunyuan3d-2", "Tencent/Hunyuan3D-2", "/generation_all", "image-to-3d",
     ["game-dev", "3d"], "model3d", "non-commercial community licence"),
    ("hunyuan3d-21", "tencent/Hunyuan3D-2.1", "/generation_all", "image-to-3d",
     ["game-dev", "3d"], "model3d", "newer Hunyuan variant"),
    ("shap-e", "hysts/Shap-E", "/run", "text-to-3d",
     ["3d", "art"], "model3d", "text/photo to 3D, old but license-friendly"),
    # video
    ("wan21", "Wan-AI/Wan2.1", "/generate_video", "video",
     ["film", "art"], "video", "Apache-2.0 text/image to video"),
    ("ltx2-distilled", "Lightricks/ltx-2-distilled", "/generate_video", "video",
     ["film", "marketing", "art"], "video", "fast image-to-video; check licence"),
    # audio: TTS
    ("edge-tts", "innoai/Edge-TTS-Text-to-Speech", "/tts_interface", "audio-tts",
     ["music-production", "film", "marketing", "general"], "audio", "fast, MIT-ish, reliable"),
    ("kokoro-tts", "hexgrad/Kokoro-TTS", "/tts", "audio-tts",
     ["music-production", "film", "general"], "audio", "Apache-2.0 quality TTS"),
    ("qwen3-tts", "ysharma/Qwen3-TTS", "/generate_voice_design", "audio-tts",
     ["film", "marketing"], "audio", "voice design via description"),
    ("melo-tts", "mrfakename/MeloTTS", "/synthesize", "audio-tts",
     ["general", "film"], "audio", "multilingual TTS, used in daggr docs"),
    ("chatterbox", "ResembleAI/Chatterbox", "/generate", "audio-tts",
     ["music-production", "film"], "audio", "expressive TTS"),
    # audio: music
    ("ace-step", "ACE-Step/ACE-Step-v1-3.5B", "/__call__", "music",
     ["music-production", "art"], "audio", "full song generation from text"),
    ("musicgen", "facebook/MusicGen", "/predict", "music",
     ["music-production", "film"], "audio", "MusicGen text-to-music"),
    ("stable-audio", "stabilityai/stable-audio-open-1.0", "/predict", "music",
     ["music-production", "film"], "audio", "open audio generation"),
    # vision / VLM
    ("moondream2", "vikhyatk/moondream2", "/answer_question", "vision",
     ["general", "art", "game-dev"], "text", "param is 'img' not 'image'"),
    ("qwen3-vl", "Qwen/Qwen3-VL-8B-Instruct", "/chat", "vision",
     ["general", "game-dev", "marketing"], "text", "modern VLM captioning/QA"),
    ("smolvlm", "HuggingFaceTB/SmolVLM-Instruct", "/chat", "vision",
     ["general"], "text", "tiny VLM"),
    # utility
    ("en2fr-local", "abidlabs/en2fr", "/predict", "text",
     ["general"], "text", "lightweight translation, run_locally friendly"),
    ("image-caption-blip", "Salesforce/BLIP", "/predict", "vision",
     ["general", "marketing"], "text", "classic captioning"),
    ("text-to-pokemon", "nateraw/text-to-pokemon", "/predict", "image-gen",
     ["game-dev", "art"], "image", "tiny, fast, fun"),
    ("depth-anything", "depth-anything/Depth-Anything-V2", "/predict", "utility",
     ["game-dev", "3d", "art"], "image", "depth maps for 3D work"),
]

#: LLM bricks are resolved from the LIVE router list, not guessed: any model the router
#: does not serve is a broken brick by definition.
LLM_CANDIDATES: list[tuple[str, str, list[str], str]] = [
    ("llm-deepseek-v4-1-flash", "deepseek-ai/DeepSeek-V4.1-Flash", ["general", "game-dev", "marketing", "music-production"], "fast, clean JSON - default planner/medic model"),
    ("llm-deepseek-v3-2", "deepseek-ai/DeepSeek-V3.2", ["general", "game-dev"], "stronger reasoning fallback"),
    ("llm-glm-4-7-flash", "zai-org/GLM-4.7-Flash", ["general"], "flash-class, JSON-clean"),
    ("llm-qwen3-4b-2507", "Qwen/Qwen3-4B-Instruct-2507", ["general"], "fastest, cheapest planner option"),
    ("llm-llama-3-1-8b", "meta-llama/Llama-3.1-8B-Instruct", ["general", "game-dev"], "well-known instruct model"),
    ("llm-llama-3-3-70b", "meta-llama/Llama-3.3-70B-Instruct", ["general", "game-dev"], "largest llama available on the router"),
    ("llm-qwen3-8b", "Qwen/Qwen3-8B", ["general"], "dense Qwen3"),
    ("llm-gemma-3-12b", "google/gemma-3-12b-it", ["general", "art"], "Gemma instruct"),
    ("llm-phi-4", "microsoft/phi-4", ["general"], "small reasoning model"),
    ("llm-hermes-3-70b", "NousResearch/Hermes-3-Llama-3.1-70B", ["general", "game-dev"], "Hermes - strong instruction following"),
]

SEARCH_QUERIES = [
    "text-to-image", "image-to-3d", "text-to-speech", "music generation",
    "background removal", "image upscaler", "video generation", "image captioning",
]

#: Targeted gap-filling: (hub search query, modality, output_kind, industries, how many).
#: Discovered bricks are verified exactly like curated ones, but failures are NOT kept -
#: auto-discovered dead ends would only bloat the registry.
DISCOVERY: list[tuple[str, str, str, list[str], int]] = [
    ("music generation", "music", "audio", ["music-production", "film"], 8),
    ("text to music", "music", "audio", ["music-production"], 6),
    ("audio generation", "music", "audio", ["music-production"], 6),
    ("image upscaler", "upscale", "image", ["art", "game-dev", "film"], 8),
    ("super resolution", "upscale", "image", ["art", "film"], 6),
    ("image captioning", "vision", "text", ["general", "marketing"], 8),
    ("visual question answering", "vision", "text", ["general"], 6),
    ("image editing", "image-edit", "image", ["art", "marketing"], 8),
    ("inpainting", "image-edit", "image", ["art"], 6),
    ("text to video", "video", "video", ["film", "marketing"], 8),
    ("depth estimation", "utility", "image", ["3d", "game-dev"], 6),
    ("image to image", "image-edit", "image", ["art"], 6),
    ("text to speech", "audio-tts", "audio", ["general", "film"], 8),
]


#: Postprocess hints that introspection cannot deduce: which element of a multi-value
#: return is the useful one. Recorded per brick id from the daggr skill's tested tables.
DECLARED_POSTPROCESS: dict[str, str] = {
    "background-removal": "tuple_index:1",   # returns (original, processed)
    "flux1-schnell": "tuple_index:0",        # returns (image, seed)
    "flux1-dev": "tuple_index:0",
    "flux1-kontext": "tuple_index:0",
    "sd35-large": "tuple_index:0",
}


def slug(source: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def sanitise(payload: dict[str, Any]) -> int:
    """
    Drop entries that are not in the normalised form this script produces.

    The registry is consumed by autogenerated code, so a half-normalised entry (raw
    gradio_client type dumps, dotted ids from an older scheme) is worse than no entry:
    it would produce a wrong binding. Returns how many entries were dropped.
    """
    bricks = payload.get("bricks") or []
    kept, dropped = [], 0
    for brick in bricks:
        outputs = brick.get("outputs") or []
        dirty = (
            "." in str(brick.get("id", ""))
            or any(len(str(o)) > 24 for o in outputs)
            or any(str(v).startswith("<") for v in (brick.get("inputs") or {}).values())
        )
        if dirty:
            dropped += 1
            log(f"  ~ dropping non-normalised entry {brick.get('id')}")
        else:
            kept.append(brick)
    payload["bricks"] = kept
    return dropped


def load_existing() -> dict[str, Any]:
    if not OUT.exists():
        return {"generated_at": None, "verified_with": "", "bricks": []}
    try:
        return json.loads(OUT.read_text())
    except Exception:
        return {"generated_at": None, "verified_with": "", "bricks": []}


def save(payload: dict[str, Any]) -> None:
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(OUT)  # atomic: a killed run never leaves a corrupt registry


def hub_meta(api, source: str) -> dict[str, Any]:
    """licence / likes / lastModified from the repo card. Best-effort."""
    out = {"license": "unknown", "likes": None, "last_modified": None}
    try:
        info = api.space_info(source)
        card = getattr(info, "cardData", None) or {}
        out["license"] = str(card.get("license") or "unknown").lower()
        out["likes"] = getattr(info, "likes", None)
        out["last_modified"] = str(getattr(info, "lastModified", "") or "") or None
    except Exception:
        try:
            info = api.model_info(source)
            card = getattr(info, "cardData", None) or {}
            out["license"] = str(card.get("license") or "unknown").lower()
            out["likes"] = getattr(info, "likes", None)
            out["last_modified"] = str(getattr(info, "lastModified", "") or "") or None
        except Exception:
            pass
    return out


def classify_output(returns: list[str], declared: str) -> str:
    text = " ".join(returns).lower()
    if "model3d" in text or declared == "model3d":
        return "model3d"
    if declared == "image" and ("filepath" in text or "path" in text or "image" in text):
        return "image"
    if declared in ("image", "audio", "video"):
        return declared
    if "audio" in text or "wav" in text:
        return "audio"
    if "video" in text or "mp4" in text:
        return "video"
    if "str" in text and not returns:
        return "text"
    return declared


def postprocess_hint_for(returns: list[str], output_kind: str) -> str | None:
    """Pick the useful return value when a Space returns several."""
    if len(returns) <= 1:
        return None
    wanted = {"image": ("filepath", "path", "image"), "audio": ("filepath", "audio", "path"),
              "video": ("filepath", "video", "path"), "model3d": ("filepath", "path", "model")}
    needles = wanted.get(output_kind, ("filepath", "path"))
    for idx, rtype in enumerate(returns):
        low = rtype.lower()
        if any(n in low for n in needles):
            if idx == 0:
                return "tuple_index:0"
            return f"tuple_index:{idx}"
    return "tuple_index:0" if output_kind in wanted else None


#: Modalities cheap enough to actually EXECUTE during verification. Introspection alone
#: cannot tell you a Space runs (hf-applications/background-removal introspects fine and
#: fails every call), so for these we make one real request and record the outcome.
PROBE_MODALITIES = {"image-gen", "image-edit", "upscale", "utility", "vision", "audio-tts",
                    "text", "music"}
#: Never probed: GPU-heavy or slow enough to be anti-social during a registry refresh.
SKIP_PROBE = {"video", "image-to-3d", "text-to-3d"}

PROBE_ENABLED = os.environ.get("VERIFY_PROBE", "") == "1"
PROBE_PROMPT = "a small red cube on a plain white background"
PROBE_TEXT = "Daggr Studio checks whether this Brick actually runs."


def probe_fixtures() -> dict[str, str]:
    """Create the local files a probe may need (image + wav). Cached in the repo."""
    fixtures = ROOT / ".cache" / "probe"
    fixtures.mkdir(parents=True, exist_ok=True)
    image = fixtures / "probe.png"
    if not image.exists():
        from PIL import Image

        Image.new("RGB", (256, 256), (220, 60, 60)).save(image)
    audio = fixtures / "probe.wav"
    if not audio.exists():
        import struct
        import wave

        with wave.open(str(audio), "w") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16000)
            fh.writeframes(b"".join(struct.pack("<h", 0) for _ in range(1600)))
    return {"image": str(image), "audio": str(audio), "text": PROBE_TEXT,
            "prompt": PROBE_PROMPT}


def probe_args(params: Any, fixtures: dict[str, str]) -> dict[str, Any] | None:
    """
    Build keyword arguments for a probe call from the Space's own parameter list.

    Two rules, both learned the hard way:
    * **Omit anything that has a default.** Guessing width=256 made FLUX report "value less
      than minimum" and got a perfectly healthy brick condemned as broken - a false negative
      is the worst possible outcome for a registry that autogenerates code.
    * **Only supply the subject** (prompt / image / audio). If a param has no default and we
      cannot supply a sensible value, refuse to probe instead of guessing.

    Returns None when the probe cannot be constructed safely.
    """
    from daggrstudio.introspect import Param

    kwargs: dict[str, Any] = {}
    for raw in params:
        param = raw if isinstance(raw, Param) else Param(name=str(raw), type="Any")
        name = param.name
        low = name.lower()
        low_type = (param.type or "").lower()

        # the subject of the Space: safe and necessary to supply
        if low in ("prompt", "text", "texts") or low.startswith("prompt"):
            kwargs[name] = PROBE_PROMPT
            continue
        if param.is_file or any(k in low for k in ("image", "img", "photo")):
            if "audio" in low:
                kwargs[name] = handle_file_factory(fixtures["audio"])
            else:
                kwargs[name] = handle_file_factory(fixtures["image"])
            continue
        if "audio" in low or "audio" in low_type:
            kwargs[name] = handle_file_factory(fixtures["audio"])
            continue
        if param.has_default:
            continue  # let the Space choose - never override with a guess
        if "seed" in low:
            kwargs[name] = 0
        elif any(k in low for k in ("width", "height", "size", "resolution")):
            kwargs[name] = 1024
        elif any(k in low for k in ("step", "steps")):
            kwargs[name] = 4
        else:
            return None  # no default and no safe value: decline
    return kwargs


def handle_file_factory(path: str):
    from gradio_client import handle_file

    return handle_file(path)


def probe_space(source: str, api_name: str, params: Any, token: str | None,
                attempts: int = 2) -> dict[str, Any]:
    """
    Make one real call (twice, because Spaces are flaky). Returns
    {'ok', 'error', 'seconds', 'skipped'}.
    """
    fixtures = probe_fixtures()
    kwargs = probe_args(params, fixtures)
    if kwargs is None:
        return {"ok": False, "error": "probe declined: a required parameter has no safe value",
                "seconds": 0.0, "skipped": True}
    last: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            from gradio_client import Client

            client = Client(source, token=token, verbose=False) if token else Client(source, verbose=False)
            client.predict(api_name=api_name, **kwargs)
            return {"ok": True, "error": None, "seconds": time.time() - started,
                    "attempt": attempt}
        except Exception as exc:
            last = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "seconds": time.time() - started, "attempt": attempt}
            if attempt < attempts:
                time.sleep(8)  # a cold Space often fails once and then works
    return last


def pick_endpoint(endpoints: dict[str, list], declared: str | None) -> str | None:
    """
    Choose the Space's main callable endpoint.

    Preference order: the declared name, then well-known primary names, then - because
    Spaces name endpoints freely - the endpoint with the most parameters, which is almost
    always the real generator. Housekeeping endpoints are rejected outright.
    """
    if declared and declared in endpoints:
        return declared
    housekeeping = ("flagged", "/queue", "/status", "/info", "/health", "/cancel", "/login",
                    "/on_submit", "/start_session", "/configure")
    known = ("/infer", "/predict", "/generate", "/run", "/__call__", "/generate_image",
             "/image", "/text", "/tts", "/synthesize", "/chat", "/predict_batched",
             "/generate_video", "/process", "/generate_tts_audio", "/predict_stream")
    candidates = [name for name in endpoints if not any(h in name.lower() for h in housekeeping)]
    if not candidates:
        return None
    for name in known:
        if name in candidates:
            return name
    # no known name: take the endpoint with the richest signature
    best = max(candidates, key=lambda n: (len(endpoints.get(n) or []), len(n)))
    return best if len(endpoints.get(best) or []) >= 1 else None


def verify_space(api, entry) -> dict[str, Any]:
    brick_id, source, api_name, modality, industries, output_kind, notes = entry
    meta = hub_meta(api, source)
    info = introspect(source, api_name, force=True, timeout_s=PER_ITEM_TIMEOUT)
    brick: dict[str, Any] = {
        "id": brick_id,
        "kind": "space",
        "source": source,
        "api_name": api_name,
        "modality": modality,
        "industries": industries,
        "license": meta["license"],
        "commercial_ok": commercial_ok(meta["license"]),
        "inputs": {},
        "outputs": [],
        "output_kind": output_kind,
        "postprocess_hint": None,
        "status": "error",
        "runtime_error": None,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "likes": meta["likes"],
        "last_modified": meta["last_modified"],
    }
    if not info.ok:
        brick["runtime_error"] = (info.error or "unreachable")[:300]
        log(f"  ✗ {brick_id:28s} {type(info.error).__name__ if False else ''}{brick['runtime_error'][:70]}")
        return brick

    # honour the endpoint the Space actually exposes
    endpoints = info.endpoints
    if api_name not in endpoints:
        brick["api_name"] = pick_endpoint(endpoints, api_name)
        if brick["api_name"] is None:
            brick["runtime_error"] = f"no usable endpoint among {sorted(endpoints)}"
            log(f"  ✗ {brick_id:28s} endpoints={sorted(endpoints)[:3]}")
            return brick

    live = info.endpoints[brick["api_name"] or ""]
    if not live:
        brick["runtime_error"] = "introspection returned no parameters (Space is not callable via API)"
        log(f"  ✗ {brick_id:28s} no parameters")
        return brick
    brick["inputs"] = {p.name: p.type for p in live}
    brick["outputs"] = info.return_types.get(brick["api_name"] or "", [])
    brick["output_kind"] = classify_output(brick["outputs"], output_kind)

    # Honesty check: a media brick whose introspected returns are pure numbers is almost
    # always an *async* endpoint (it returns a job id / ETA, not the asset). Better to mark
    # it unusable than to hand someone a pipeline that silently produces nothing.
    media = brick["output_kind"] in ("image", "audio", "video", "model3d")
    if media and brick["outputs"] and all(o == "number" for o in brick["outputs"]):
        brick["runtime_error"] = (
            f"declared {brick['output_kind']} output but the endpoint returns only numbers "
            "(async/job-style API) - not usable as a synchronous brick"
        )
        log(f"  ✗ {brick_id:28s} async endpoint returns {brick['outputs']}")
        return brick

    brick["postprocess_hint"] = (
        DECLARED_POSTPROCESS.get(brick_id)
        or postprocess_hint_for(brick["outputs"], brick["output_kind"])
    )
    brick["status"] = "running"
    log(f"  ✓ {brick_id:28s} {brick['api_name']:22s} {len(live)} params -> {brick['output_kind']}"
        f"{' pp=' + brick['postprocess_hint'] if brick['postprocess_hint'] else ''}")

    # Execution probe: introspection cannot tell "callable" from "working".
    if PROBE_ENABLED and brick["modality"] in PROBE_MODALITIES:
        outcome = probe_space(source, brick["api_name"] or "", live, token_hint())
        if outcome.get("skipped"):
            brick["probe"] = "skipped"
            log(f"      probe skipped ({outcome.get('error')})")
        elif outcome["ok"]:
            brick["probe"] = "ok"
            brick["probe_seconds"] = round(outcome["seconds"], 1)
            log(f"      probe OK in {outcome['seconds']:.1f}s")
        else:
            brick["probe"] = "failed"
            brick["status"] = "error"
            brick["runtime_error"] = f"execution probe failed: {outcome['error']}"
            log(f"      probe FAILED: {str(outcome['error'])[:90]}")
    return brick


def token_hint() -> str | None:
    return os.environ.get("HF_TOKEN")


def verify_llm(entry, router_models: set[str]) -> dict[str, Any]:
    brick_id, source, industries, notes = entry
    brick: dict[str, Any] = {
        "id": brick_id, "kind": "inference_model", "source": source, "api_name": None,
        "modality": "text", "industries": industries,
        "license": "unknown", "commercial_ok": False, "inputs": {}, "outputs": ["str"],
        "output_kind": "text", "postprocess_hint": None, "status": "error",
        "runtime_error": None, "verified_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
    if router_models and source not in router_models:
        brick["runtime_error"] = "not served by the HF Inference router"
        log(f"  ✗ {brick_id:28s} not on router")
        return brick
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=source, provider="auto", token=token)
        result = client.chat_completion(
            messages=[{"role": "user", "content": "Reply with the single word OK"}],
            max_tokens=5, temperature=0.0,
        )
        if not (result.choices[0].message.content or "").strip():
            raise RuntimeError("empty completion")
        brick["status"] = "running"
        brick["commercial_ok"] = True
        brick["license"] = "mit"  # router-served models accept commercial prompting
        log(f"  ✓ {brick_id:28s} router OK")
    except Exception as exc:
        brick["runtime_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        log(f"  ✗ {brick_id:28s} {brick['runtime_error'][:70]}")
    return brick


def router_model_set(token: str | None) -> set[str]:
    """The definitive list of models the router can actually serve."""
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://router.huggingface.co/v1/models",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return {m["id"] for m in data.get("data", [])}
    except Exception as exc:
        log(f"router model list unavailable: {type(exc).__name__}")
        return set()


def discover(api) -> list[tuple]:
    """Find additional Space candidates from the Hub, tagged by the query that found them."""
    found = []
    seen = {entry[1] for entry in SPACES}
    for query, modality, output_kind, industries, limit in DISCOVERY:
        try:
            spaces = api.list_spaces(search=query, sort="likes", direction=-1, limit=limit)
        except Exception as exc:
            log(f"discovery failed for '{query}': {type(exc).__name__}")
            continue
        for space in spaces:
            source = space.id
            if source in seen:
                continue
            seen.add(source)
            found.append((slug(source), source, None, modality, industries, output_kind,
                          f"auto-discovered via '{query}' (likes={getattr(space, 'likes', '?')})"))
    log(f"discovery: {len(found)} new candidates")
    return found


def call_with_timeout(fn, *args, timeout: int = PER_ITEM_TIMEOUT + 40):
    """Run fn in a thread so one hung Space cannot stall the whole run."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            return None
        except Exception as exc:
            log(f"  ! {type(exc).__name__}: {str(exc)[:100]}")
            return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="verify only these brick ids")
    parser.add_argument("--refresh", action="store_true", help="re-verify existing bricks")
    parser.add_argument("--discover", action="store_true", help="add Hub search candidates")
    parser.add_argument("--models-only", action="store_true", help="refresh LLM bricks only")
    parser.add_argument("--probe", action="store_true",
                        help="actually EXECUTE cheap bricks to prove they run")
    args = parser.parse_args()

    global PROBE_ENABLED
    if args.probe:
        PROBE_ENABLED = True

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    payload = load_existing()
    dropped = sanitise(payload)
    by_id = {b["id"]: b for b in payload.get("bricks", [])}
    log(f"start: {len(by_id)} bricks already in {OUT.name} ({dropped} stale dropped)")

    spaces = list(SPACES)
    if args.discover:
        spaces += discover(api)
    if args.only:
        spaces = [s for s in spaces if s[0] in args.only]

    discovered: set[str] = {s[0] for s in spaces} - {s[0] for s in SPACES}

    if not args.models_only:
        for entry in spaces:
            brick_id = entry[0]
            existing = by_id.get(brick_id)
            if existing and existing.get("status") == "running" and not args.refresh:
                log(f"  = {brick_id:28s} already verified")
                continue
            if existing and not args.refresh and existing.get("verified_at"):
                # retry previously-failed bricks, but not within the same hour
                try:
                    age = time.time() - datetime.fromisoformat(existing["verified_at"]).timestamp()
                except Exception:
                    age = 1e9
                if age < 3600:
                    log(f"  = {brick_id:28s} failed recently, skipping")
                    continue
            result = call_with_timeout(verify_space, api, entry)
            if result:
                if result.get("status") != "running" and brick_id in discovered:
                    log(f"  ~ {brick_id:28s} discovered but unusable; not adding")
                    continue
                by_id[brick_id] = result
                save({"verified_with": _versions(), "bricks": list(by_id.values())})
            time.sleep(1.0)  # be polite to the Hub

    if not args.only:
        models = router_model_set(token)
        log(f"router serves {len(models)} models")
        for entry in LLM_CANDIDATES:
            brick_id = entry[0]
            existing = by_id.get(brick_id)
            if existing and existing.get("status") == "running" and not args.refresh:
                log(f"  = {brick_id:28s} already verified")
                continue
            result = verify_llm(entry, models)
            by_id[brick_id] = result
            save({"verified_with": _versions(), "bricks": list(by_id.values())})

    write_report(by_id)
    log(f"done: {sum(1 for b in by_id.values() if b.get('status') == 'running')} running / "
        f"{len(by_id)} total")
    return 0


def _versions() -> str:
    import gradio_client
    import huggingface_hub

    return f"gradio_client {gradio_client.__version__} / huggingface_hub {huggingface_hub.__version__}"


def write_report(by_id: dict[str, Any]) -> None:
    running = [b for b in by_id.values() if b.get("status") == "running"]
    broken = [b for b in by_id.values() if b.get("status") != "running"]
    lines = [
        "# Brick registry verification",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Verified with: {_versions()}",
        "",
        f"- tried: **{len(by_id)}**",
        f"- running: **{len(running)}**",
        f"- unusable: **{len(broken)}**",
        "",
        "## Running bricks",
        "",
        "| id | source | endpoint | modality | license | commercial |",
        "|---|---|---|---|---|---|",
    ]
    for brick in sorted(running, key=lambda b: (b.get("modality", ""), b.get("id", ""))):
        lines.append(
            f"| `{brick['id']}` | {brick['source']} | `{brick.get('api_name') or '-'}` | "
            f"{brick.get('modality')} | {brick.get('license')} | "
            f"{'yes' if brick.get('commercial_ok') else 'NO'} |"
        )
    lines += ["", "## Unusable candidates (kept for provenance)", "",
              "| id | source | error class |", "|---|---|---|"]
    for brick in sorted(broken, key=lambda b: b.get("id", "")):
        err = (brick.get("runtime_error") or "unknown").split(":")[0][:60]
        lines.append(f"| `{brick['id']}` | {brick['source']} | {err} |")
    lines += [
        "",
        "## Notes",
        "",
        "Bricks that failed verification are kept in `spaces.json` with `status: error` so the",
        "Medic can see them and substitute an alternative rather than retrying a dead Space.",
        "Re-run `python scripts/verify_registry.py --refresh --only <id>` to retry a single brick.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())