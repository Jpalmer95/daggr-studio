#!/usr/bin/env python
"""
Create/update the Daggr Studio Space and wait for it to come up.

    python scripts/deploy_space.py                       # deploy jkorstad/daggr-studio
    python scripts/deploy_space.py --repo jkorstad/x --hardware cpu-basic
    python scripts/deploy_space.py --poll-only           # just watch the runtime

The Space runs the `docker` SDK (we control Python + gradio versions), on CPU only - it
orchestrates other Spaces, it never runs GPU models itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Everything the Space needs at runtime. Deliberately excludes the local venv, output
#: artifacts, docs and the registry/verification logs (the JSON registry IS included).
INCLUDE_FILES = ["app.py", "requirements.txt", "Dockerfile", "README.md", "pytest.ini",
                 "LICENSE", "MASTER_PLAN.md"]
INCLUDE_DIRS = ["daggrstudio", "scripts", "tests"]


def token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    cached = Path.home() / ".cache" / "huggingface" / "token"
    return cached.read_text().strip() if cached.exists() else None


def iter_files():
    for name in INCLUDE_FILES:
        path = ROOT / name
        if path.exists():
            yield path, name
    for dirname in INCLUDE_DIRS:
        base = ROOT / dirname
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if "__pycache__" in rel or rel.endswith(".pyc"):
                continue
            if dirname == "scripts" and path.name.startswith(("smoke", "acceptance")):
                continue  # dev-only scripts; not needed in the Space image
            yield path, rel


def poll(api, repo: str, timeout: int = 900) -> str:
    started = time.time()
    last = ""
    while time.time() - started < timeout:
        try:
            info = api.space_info(repo, expand=["runtime"])
            stage = str(getattr(info.runtime, "stage", "") if info.runtime else "")
        except Exception as exc:
            stage = f"unknown ({type(exc).__name__})"
        if stage != last:
            print(f"  [{time.time() - started:5.0f}s] {stage}", flush=True)
            last = stage
        if stage in ("RUNNING", "RUNNING_BUILDING"):
            return stage
        if stage in ("BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR", "DELETED"):
            raw = getattr(info.runtime, "raw", {}) or {}
            print(f"\nSpace failed to start: {stage}")
            print(f"  error: {str(raw.get('errorMessage'))[:1500]}")
            return stage
        time.sleep(15)
    return "TIMEOUT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("DAGGRSTUDIO_SPACE",
                                                         "jkorstad/daggr-studio"))
    parser.add_argument("--hardware", default="cpu-basic")
    parser.add_argument("--poll-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    tok = token()
    if not tok:
        print("No HF token available; set HF_TOKEN.")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    user = api.whoami().get("name")
    repo = args.repo if "/" in args.repo else f"{user}/{args.repo}"

    if not args.poll_only:
        print(f"creating/updating space {repo} (docker, {args.hardware})")
        api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker",
                        space_hardware=args.hardware, exist_ok=True)
        uploaded = 0
        for path, rel in iter_files():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=rel, repo_id=repo,
                            repo_type="space",
                            commit_message=f"deploy: {rel}")
            uploaded += 1
        print(f"uploaded {uploaded} files")
        try:
            api.add_space_secret(repo_id=repo, key="HF_TOKEN", value=tok)
            print("set HF_TOKEN secret (used only for the capped community pool)")
        except Exception as exc:
            print(f"could not set the secret: {type(exc).__name__}: {exc}")

    print("waiting for the Space to come up…")
    stage = poll(api, repo, timeout=args.timeout)
    url = f"https://huggingface.co/spaces/{repo}"
    print(f"\n{stage}  {url}")
    if stage in ("RUNNING", "RUNNING_BUILDING"):
        host = url.replace("huggingface.co/spaces/", "").replace("/", "-") + ".hf.space"
        host = host.replace("https://", "")
        print(f"  app: https://{host}")
        print(f"  builder: https://{host}/builder")
        print(f"  health: https://{host}/healthz")
    return 0 if stage in ("RUNNING", "RUNNING_BUILDING") else 1


if __name__ == "__main__":
    raise SystemExit(main())