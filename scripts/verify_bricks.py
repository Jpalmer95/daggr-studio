#!/usr/bin/env python3
"""
Verify bricks for daggr-studio registry.
Phased approach:
  1) Search HF Hub for candidate Spaces + models per modality.
  2) Verify each candidate: existence, license, likes, lastModified, runtime stage.
  3) Call gradio_client.Client.view_api() for Spaces (or InferenceClient.chat_completion for models).
  4) Write verified entries to JSON incrementally.
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import Any

# Ensure venv packages
sys.path.insert(0, "/home/jonathan/dev/daggr-pipelines/.venv/lib/python3.12/site-packages")

from huggingface_hub import HfApi, InferenceClient
from gradio_client import Client as GradioClient

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    # Try extracting from bashrc without printing
    try:
        with open(os.path.expanduser("~/.bashrc")) as f:
            for line in f:
                if "HF_TOKEN=" in line:
                    HF_TOKEN = line.split('"')[1]
                    break
    except Exception:
        pass

api = HfApi(token=HF_TOKEN)

OUT_PATH = "/home/jonathan/dev/daggr-studio/daggrstudio/registry/spaces.json"
LOG_PATH = "/home/jonathan/dev/daggr-studio/daggrstudio/registry/verification_log.jsonl"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- helpers ---

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def kebab_id(source: str, api_name: str = None) -> str:
    base = source.replace("/", "-")
    if api_name and api_name != "/" and api_name != "/predict" and api_name != "/infer":
        base += api_name.replace("/", "-")
    return base.lower()

def license_to_commercial_ok(lic: str) -> bool:
    if not lic:
        return False
    lic = lic.lower().strip()
    ok_licenses = ["apache-2.0", "mit", "bsd", "openrail", "cc-by-4.0", "creativeml-openrail-m"]
    bad = ["nc", "non-commercial", "cc-by-nc", "cc-by-sa", "gpl", "unknown", "flux-dev"]
    if any(b in lic for b in bad):
        return False
    return any(o in lic for o in ok_licenses)

def modality_from_category(cat: str) -> str:
    mapping = {
        "image-gen": "image-gen",
        "image-edit": "image-edit",
        "3d": "image-to-3d",
        "video": "video",
        "audio": "music",
        "tts": "audio-tts",
        "vision": "vision",
        "text": "text",
        "upscale": "upscale",
        "utility": "utility",
        "translation": "utility",
    }
    return mapping.get(cat, "utility")

def output_kind_from_types(types: list) -> str:
    """Infer output kind from gradio python type strings."""
    if not types:
        return "text"
    t = " ".join(str(x) for x in types).lower()
    if "filepath" in t or "file" in t or ".glb" in t or ".obj" in t:
        if ".glb" in t or ".obj" in t or "3d" in t:
            return "model3d"
        if ".wav" in t or ".mp3" in t or "audio" in t:
            return "audio"
        if ".mp4" in t or ".mov" in t or "video" in t:
            return "video"
        return "image"
    if "image" in t or "pil" in t or "np.array" in t:
        return "image"
    if "audio" in t or "filepath" in t:
        return "audio"
    if "str" in t:
        return "text"
    return "text"

def industries_from_modality(mod: str) -> list:
    m = {
        "image-gen": ["art", "game-dev", "marketing", "general"],
        "image-edit": ["art", "game-dev", "marketing", "general"],
        "image-to-3d": ["3d", "game-dev", "art"],
        "text-to-3d": ["3d", "game-dev", "art"],
        "video": ["film", "marketing", "game-dev", "general"],
        "audio-tts": ["game-dev", "marketing", "general"],
        "music": ["music-production", "game-dev", "marketing"],
        "vision": ["general", "game-dev", "marketing"],
        "upscale": ["art", "game-dev", "film", "general"],
        "text": ["general", "marketing", "game-dev"],
        "utility": ["general"],
    }
    return m.get(mod, ["general"])

# --- Search candidates ---

def search_spaces_by_query(query: str, limit: int = 15):
    """Search HF Spaces and return candidates."""
    results = []
    try:
        spaces = api.list_spaces(search=query, sort="likes", limit=limit)
        for s in spaces:
            results.append({
                "id": s.id,
                "likes": getattr(s, "likes", 0),
                "lastModified": getattr(s, "lastModified", None),
            })
    except Exception as e:
        print(f"[search error] {query}: {e}")
    return results

def search_models_by_query(query: str, limit: int = 15, pipeline_tag: str = None):
    results = []
    try:
        kwargs = {"search": query, "sort": "likes", "limit": limit}
        if pipeline_tag:
            kwargs["pipeline_tag"] = pipeline_tag
        models = api.list_models(**kwargs)
        for m in models:
            results.append({
                "id": m.id,
                "likes": getattr(m, "likes", 0),
                "lastModified": getattr(m, "lastModified", None),
            })
    except Exception as e:
        print(f"[search error] {query}: {e}")
    return results

# --- Verification core ---

verified_bricks = []
failed_candidates = []

def verify_space(source: str, expected_api: str = None, category_hint: str = None) -> dict:
    """Verify a Space. Returns brick dict or None on failure."""
    result = {
        "id": kebab_id(source, expected_api),
        "kind": "space",
        "source": source,
        "api_name": expected_api,
        "modality": modality_from_category(category_hint or "utility"),
        "industries": industries_from_modality(modality_from_category(category_hint or "utility")),
        "license": "unknown",
        "commercial_ok": False,
        "inputs": {},
        "outputs": [],
        "output_kind": "text",
        "postprocess_hint": None,
        "status": "error",
        "runtime_error": None,
        "verified_at": now_iso(),
        "notes": "",
        "likes": 0,
        "lastModified": None,
    }

    # 1. Space existence + metadata
    try:
        info = api.space_info(source)
        result["license"] = (info.card_data or {}).get("license", "unknown") if info.card_data else "unknown"
        result["commercial_ok"] = license_to_commercial_ok(result["license"])
        result["likes"] = getattr(info, "likes", 0)
        result["lastModified"] = getattr(info, "lastModified", None)
        if result["lastModified"]:
            try:
                result["lastModified"] = result["lastModified"].isoformat()
            except Exception:
                pass
    except Exception as e:
        result["runtime_error"] = f"space_info failed: {type(e).__name__}: {str(e)[:120]}"
        failed_candidates.append({"source": source, "error": result["runtime_error"], "kind": "space"})
        return result

    # 2. Runtime stage
    try:
        info_runtime = api.space_info(source, expand=["runtime"])
        runtime = info_runtime.runtime
        stage = "UNKNOWN"
        if runtime:
            raw = runtime.raw if hasattr(runtime, "raw") else {}
            if isinstance(raw, dict):
                stage = raw.get("stage", "UNKNOWN")
        result["status"] = stage.lower() if stage else "unknown"
        if stage not in ("RUNNING", "RUNNING_BUILDING"):
            result["notes"] += f"Stage={stage}; "
    except Exception as e:
        result["notes"] += f"runtime check failed: {type(e).__name__}; "

    # 3. Gradio API introspection
    try:
        client = GradioClient(source, token=HF_TOKEN)
        api_info = client.view_api(return_format="dict", print_info=False)
        endpoints = api_info.get("named_endpoints", {}) or api_info.get("unnamed_endpoints", {})

        # Pick endpoint
        chosen_endpoint = None
        if expected_api and expected_api in endpoints:
            chosen_endpoint = expected_api
        else:
            # Heuristic: pick endpoint with most inputs or most likes in name
            for ep_name, ep_data in endpoints.items():
                if ep_data and ep_data.get("parameters"):
                    chosen_endpoint = ep_name
                    break
            if not chosen_endpoint and endpoints:
                chosen_endpoint = list(endpoints.keys())[0]

        if chosen_endpoint:
            result["api_name"] = chosen_endpoint
            result["id"] = kebab_id(source, chosen_endpoint)
            ep_data = endpoints.get(chosen_endpoint, {})
            params = ep_data.get("parameters", [])
            returns = ep_data.get("returns", [])

            inputs = {}
            for p in params:
                pname = p.get("parameter_name", p.get("name", "unknown"))
                ptype_obj = p.get("python_type", {})
                ptype = ptype_obj.get("type", "str") if isinstance(ptype_obj, dict) else str(ptype_obj)
                inputs[pname] = ptype
            result["inputs"] = inputs

            outputs = []
            for r in returns:
                rtype_obj = r.get("python_type", {})
                rtype = rtype_obj.get("type", "str") if isinstance(rtype_obj, dict) else str(rtype_obj)
                outputs.append(rtype)
            result["outputs"] = outputs
            result["output_kind"] = output_kind_from_types(outputs)

            # Postprocess hint if tuple with >1 element and first is not the useful one
            if len(outputs) > 1:
                # Common pattern: (original, processed) -> processed is index 1
                if any("image" in o.lower() or "filepath" in o.lower() for o in outputs[1:]):
                    result["postprocess_hint"] = "tuple_index:1"
        else:
            result["runtime_error"] = "No callable endpoint found in view_api()"
            result["status"] = "error"

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
        result["runtime_error"] = err
        result["status"] = "error"
        failed_candidates.append({"source": source, "error": err, "kind": "space"})

    return result


def verify_inference_model(model_id: str, category_hint: str = "text") -> dict:
    """Verify an HF Inference model via chat_completion."""
    result = {
        "id": kebab_id(model_id),
        "kind": "inference_model",
        "source": model_id,
        "api_name": None,
        "modality": modality_from_category(category_hint),
        "industries": industries_from_modality(modality_from_category(category_hint)),
        "license": "unknown",
        "commercial_ok": False,
        "inputs": {},
        "outputs": ["str"],
        "output_kind": "text",
        "postprocess_hint": None,
        "status": "error",
        "runtime_error": None,
        "verified_at": now_iso(),
        "notes": "",
        "likes": 0,
        "lastModified": None,
    }

    # 1. Model existence
    try:
        info = api.model_info(model_id)
        result["license"] = (info.card_data or {}).get("license", "unknown") if info.card_data else "unknown"
        result["commercial_ok"] = license_to_commercial_ok(result["license"])
        result["likes"] = getattr(info, "likes", 0)
        result["lastModified"] = getattr(info, "lastModified", None)
        if result["lastModified"]:
            try:
                result["lastModified"] = result["lastModified"].isoformat()
            except Exception:
                pass
    except Exception as e:
        result["runtime_error"] = f"model_info failed: {type(e).__name__}: {str(e)[:120]}"
        failed_candidates.append({"source": model_id, "error": result["runtime_error"], "kind": "inference_model"})
        return result

    # 2. InferenceClient.chat_completion
    try:
        client = InferenceClient(model=model_id, provider="auto", token=HF_TOKEN)
        # Use a tiny prompt that should work for chat models
        resp = client.chat_completion(
            messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
            max_tokens=5,
            temperature=0.1,
        )
        if resp and resp.choices:
            result["status"] = "running"
            result["notes"] = "Verified via InferenceClient.chat_completion"
        else:
            result["runtime_error"] = "Empty response from chat_completion"
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
        result["runtime_error"] = err
        result["status"] = "error"
        failed_candidates.append({"source": model_id, "error": err, "kind": "inference_model"})

    return result


def save_progress():
    payload = {
        "generated_at": now_iso(),
        "verified_with": "gradio_client 2.5.0 / huggingface_hub 1.17.0",
        "bricks": verified_bricks,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(LOG_PATH, "a") as f:
        for b in verified_bricks:
            f.write(json.dumps({"source": b["source"], "status": b["status"], "verified_at": b["verified_at"]}) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["search", "verify", "all"], default="all")
    parser.add_argument("--only", help="comma-separated list of space/model IDs to verify")
    args = parser.parse_args()

    if args.phase in ("search", "all"):
        print("=== Searching for additional candidates ===")
        queries = [
            ("text to image", 10),
            ("background removal", 10),
            ("image to 3D", 10),
            ("text to video", 10),
            ("text to speech", 10),
            ("music generation", 10),
            ("upscale image", 10),
            ("vision language model", 10),
        ]
        for q, lim in queries:
            print(f"\n-- Searching Spaces: {q}")
            for s in search_spaces_by_query(q, lim):
                print(f"   {s['id']} | likes={s['likes']}")
            print(f"-- Searching Models: {q}")
            for m in search_models_by_query(q, lim):
                print(f"   {m['id']} | likes={m['likes']}")

    if args.phase in ("verify", "all"):
        # Define all candidates from known registries + new search
        # These are our primary candidates
        candidates = [
            # image-gen (>=4)
            ("space", "black-forest-labs/FLUX.1-schnell", "/infer", "image-gen"),
            ("space", "black-forest-labs/FLUX.1-dev", "/infer", "image-gen"),
            ("space", "stabilityai/stable-diffusion-3.5-large", "/infer", "image-gen"),
            ("space", "hf-applications/Z-Image-Turbo", "/generate_image", "image-gen"),
            ("space", "radames/Real-Time-Text-to-Image-SDXL-Lightning", None, "image-gen"),  # NEW search

            # image-edit (>=2)
            ("space", "hf-applications/background-removal", "/image", "image-edit"),
            ("space", "not-lain/background-removal", "/run", "image-edit"),
            ("space", "briaai/RMBG-2-Studio", None, "image-edit"),  # NEW

            # upscale / utility (>=2)
            ("space", "akhaliq/Real-ESRGAN", None, "upscale"),  # NEW
            ("space", "nightfury/Image_Face_Upscale_Restoration-GFPGAN", None, "upscale"),  # NEW search

            # 3d (>=3)
            ("space", "VAST-AI/TripoSG", "/generate", "3d"),
            ("space", "Tencent/Hunyuan3D-2", "/generation_all", "3d"),
            ("space", "JeffreyXiang/TRELLIS", "/generate", "3d"),
            ("space", "facebook/shap-e", None, "3d"),  # NEW

            # video (>=2)
            ("space", "Wan-AI/Wan2.1", "/generate", "video"),
            ("space", "ali-vilab/modelscope-text-to-video-synthesis", None, "video"),  # NEW search

            # tts (>=4)
            ("space", "hexgrad/Kokoro-TTS", "/tts", "tts"),
            ("space", "innoai/Edge-TTS-Text-to-Speech", "/tts_interface", "tts"),
            ("space", "ysharma/Qwen3-TTS", "/generate_voice_design", "tts"),
            ("space", "NihalGazi/Text-To-Speech-Unlimited", None, "tts"),  # NEW search

            # music (>=2)
            ("space", "facebook/audiocraft", None, "audio"),  # music
            ("space", "suno/bark", None, "audio"),  # music / audio gen

            # vision (>=2)
            ("space", "vikhyatk/moondream2", "/answer_question", "vision"),
            ("space", "nvidia/NVLM", "/img_chat", "vision"),

            # text / LLM inference models (>=5)
            ("inference", "google/gemma-2-2b-it", "text"),
            ("inference", "meta-llama/Llama-3.2-1B-Instruct", "text"),
            ("inference", "Qwen/Qwen2.5-7B-Instruct", "text"),
            ("inference", "microsoft/Phi-3-mini-4k-instruct", "text"),
            ("inference", "mistralai/Mistral-7B-Instruct-v0.3", "text"),
        ]

        if args.only:
            only_set = set(x.strip() for x in args.only.split(","))
            candidates = [c for c in candidates if c[1] in only_set]

        print(f"=== Verifying {len(candidates)} candidates ===")
        for kind, source, api_or_tag, *extra in candidates:
            category = extra[0] if extra else api_or_tag
            if kind == "space":
                print(f"\n[SPACE] {source}  (expected api={api_or_tag})")
                brick = verify_space(source, expected_api=api_or_tag, category_hint=category)
            else:
                print(f"\n[MODEL] {source}")
                brick = verify_inference_model(source, category_hint=api_or_tag)

            print(f"  -> status={brick['status']}  license={brick['license']}  commercial_ok={brick['commercial_ok']}")
            if brick["runtime_error"]:
                print(f"  -> ERROR: {brick['runtime_error']}")
            if brick["inputs"]:
                print(f"  -> inputs={brick['inputs']}")
            if brick["outputs"]:
                print(f"  -> outputs={brick['outputs']}")

            verified_bricks.append(brick)
            save_progress()

        print(f"\n=== DONE ===")
        print(f"Verified bricks: {len(verified_bricks)}")
        print(f"Failed candidates logged: {len(failed_candidates)}")
        print(f"Output: {OUT_PATH}")
