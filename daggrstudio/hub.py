"""
Sharing layer: a Hugging Face Dataset as the database.

Why a Dataset and not a server:
* it is free and needs no infrastructure to operate;
* every published workflow is a commit - auditable, diffable, forkable, and removable;
* reads are public, so browsing the leaderboard costs nothing for anyone (see MASTER_PLAN §3);
* writes use the *user's* token (BYOK), so publishing scales with contributors, not with us.

Layout inside the dataset repo::

    index.json                  # small aggregate record per workflow (fast listing)
    workflows/<slug>.json       # the full spec + metadata + provenance
    votes/<slug>.json           # {voters: {username: +1|-1}, ...}  (one file per workflow)

Trade-off, stated plainly: there are no atomic counters. Two people voting at the same
instant can lose a vote (last write wins). At community scale that is an acceptable price
for zero infrastructure, and the vote files stay human-readable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daggrstudio.spec import WorkflowSpec, slugify

DATASET_ID = os.environ.get("DAGGRSTUDIO_DATASET", "jkorstad/daggr-studio-workflows")
INDEX_FILE = "index.json"
LOCAL_MIRROR = Path(os.environ.get("DAGGRSTUDIO_CACHE",
                                   str(Path.home() / ".cache" / "daggr-studio"))) / "workflows"
LIST_TTL = 300  # seconds; keeps the leaderboard snappy without hammering the Hub

MODALITY_LABELS = {
    "text": "Text / LLM",
    "image-gen": "Image generation",
    "image-edit": "Image editing",
    "image-to-3d": "Image to 3D",
    "text-to-3d": "Text to 3D",
    "video": "Video",
    "audio-tts": "Speech",
    "music": "Music",
    "vision": "Vision",
    "upscale": "Upscaling",
    "utility": "Utility",
}


class ShareError(RuntimeError):
    """Raised when a publish/vote cannot be completed (usually: no write token)."""


@dataclass
class StoreResult:
    ok: bool
    message: str = ""
    url: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "message": self.message, "url": self.url, "data": self.data}


# ─── metadata derivation ──────────────────────────────────────────────────────


def spec_metadata(
    spec: WorkflowSpec,
    author: str = "",
    username: str = "",
    registry: Any | None = None,
) -> dict[str, Any]:
    """Everything the leaderboard needs, derived from the spec (never hand-entered)."""
    from daggrstudio.registry.bricks import get_registry

    reg = registry or get_registry()
    modalities = sorted({
        brick.modality
        for brick in (reg.by_id(b) for b in spec.brick_ids)
        if brick is not None
    })
    return {
        "slug": spec.slug,
        "name": spec.name,
        "intent": spec.intent,
        "author": author or username or spec.author or "anonymous",
        "hf_user": username,
        "industry": spec.industry,
        "tags": spec.tags,
        "license_posture": spec.license_posture,
        "compute_tier": spec.compute_tier,
        "brick_ids": spec.brick_ids,
        "steps": len(spec.steps),
        "modalities": modalities,
        "created_at": spec.created_at,
        "updated_at": spec.updated_at,
        "planner_model": spec.planner_model,
        "heal_rounds": len(spec.heal_log),
        "schema_version": spec.schema_version,
        "votes": 0,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _path(slug: str) -> str:
    return f"workflows/{slugify(slug)}.json"


def _votes_path(slug: str) -> str:
    return f"votes/{slugify(slug)}.json"


# ─── local mirror (lets the app work without any token) ───────────────────────


def mirror_write(entry: dict[str, Any]) -> None:
    try:
        LOCAL_MIRROR.mkdir(parents=True, exist_ok=True)
        (LOCAL_MIRROR / f"{entry['slug']}.json").write_text(json.dumps(entry, indent=2))
    except Exception:
        pass


def mirror_read_all() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not LOCAL_MIRROR.exists():
        return out
    for path in sorted(LOCAL_MIRROR.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except Exception:
            continue
    return out


# ─── hub access ───────────────────────────────────────────────────────────────


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    return HfApi(token=token or os.environ.get("HF_TOKEN") or None)


def dataset_url(slug: str = "") -> str:
    base = f"https://huggingface.co/datasets/{DATASET_ID}"
    return f"{base}/blob/main/{_path(slug)}" if slug else base


def ensure_dataset(token: str | None) -> StoreResult:
    if not token:
        return StoreResult(False, "publishing needs your own Hugging Face token "
                                  "(Settings → token), so commits are attributed to you")
    try:
        api = _api(token)
        api.create_repo(repo_id=DATASET_ID, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=json.dumps({
                "dataset": DATASET_ID,
                "purpose": "Daggr Studio shared workflows + votes. Each file is one workflow.",
                "schema_version": 1,
                "license": "cc-by-4.0 for the workflow metadata; the workflow keeps its own "
                           "licence posture for the assets it produces.",
            }, indent=2).encode(),
            path_in_repo="README.json",
            repo_id=DATASET_ID,
            repo_type="dataset",
            commit_message="chore: dataset descriptor",
        )
        return StoreResult(True, f"dataset ready: {dataset_url()}", url=dataset_url())
    except Exception as exc:
        return StoreResult(False, f"could not prepare the shared dataset: "
                                  f"{type(exc).__name__}: {str(exc)[:200]}")


def _read_remote_json(filename: str, token: str | None = None) -> dict[str, Any] | None:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=DATASET_ID, filename=filename, repo_type="dataset",
                               token=token or None, force_download=True)
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def publish(
    spec: WorkflowSpec,
    token: str | None = None,
    username: str = "",
    author: str = "",
    note: str = "",
) -> StoreResult:
    """
    Publish (or update) one workflow. BYOK: without a token we still save it locally so the
    user can export the file, but we say clearly that it was not shared.
    """
    entry = spec_metadata(spec, author=author, username=username)
    entry["note"] = note[:400]
    entry["spec"] = spec.to_dict()
    mirror_write(entry)

    if not token:
        return StoreResult(
            False,
            "saved to this Space only. Add your Hugging Face token in Settings to publish it "
            "to the shared leaderboard.",
            data=entry,
        )

    prepared = ensure_dataset(token)
    if not prepared.ok:
        return StoreResult(False, prepared.message, data=entry)

    api = _api(token)
    try:
        api.upload_file(
            path_or_fileobj=json.dumps(entry, indent=2).encode(),
            path_in_repo=_path(spec.slug),
            repo_id=DATASET_ID,
            repo_type="dataset",
            commit_message=f"workflow: {spec.name} ({spec.slug}) by {entry['author']}",
        )
        index = _read_remote_json(INDEX_FILE, token) or {"workflows": {}}
        summary = {k: v for k, v in entry.items() if k != "spec"}
        votes = _read_remote_json(_votes_path(spec.slug), token) or {"voters": {}}
        summary["votes"] = sum(1 for v in (votes.get("voters") or {}).values() if v > 0)
        index.setdefault("workflows", {})[spec.slug] = summary
        index["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        api.upload_file(
            path_or_fileobj=json.dumps(index, indent=2).encode(),
            path_in_repo=INDEX_FILE,
            repo_id=DATASET_ID,
            repo_type="dataset",
            commit_message=f"index: {spec.slug}",
        )
        return StoreResult(True, f"published '{spec.name}' to the shared leaderboard",
                           url=dataset_url(spec.slug), data=entry)
    except Exception as exc:
        return StoreResult(False, f"publishing failed: {type(exc).__name__}: {str(exc)[:200]}",
                           data=entry)


def load(slug: str, token: str | None = None) -> StoreResult:
    """Load one workflow by slug: mirror first (fast, offline), then the Hub."""
    slug = slugify(slug)
    local = LOCAL_MIRROR / f"{slug}.json"
    if local.exists():
        try:
            payload = json.loads(local.read_text())
            return StoreResult(True, f"loaded '{slug}' from this Space",
                               data=payload)
        except Exception:
            pass
    payload = _read_remote_json(_path(slug), token)
    if not payload:
        return StoreResult(False, f"no workflow named '{slug}' found")
    return StoreResult(True, f"loaded '{slug}' from the shared dataset", data=payload)


def spec_from_entry(entry: dict[str, Any]) -> WorkflowSpec | None:
    payload = entry.get("spec") if isinstance(entry, dict) else None
    if not isinstance(payload, dict):
        return None
    spec = WorkflowSpec.from_dict(payload)
    spec.author = entry.get("author", spec.author)
    return spec


def list_workflows(token: str | None = None, include_mirror: bool = True) -> list[dict[str, Any]]:
    """Leaderboard rows: remote index merged with anything saved locally."""
    rows: dict[str, dict[str, Any]] = {}
    index = _read_remote_json(INDEX_FILE, token)
    if isinstance(index, dict):
        for slug, summary in (index.get("workflows") or {}).items():
            if isinstance(summary, dict):
                rows[slug] = summary
    if include_mirror:
        for entry in mirror_read_all():
            slug = entry.get("slug")
            if not slug:
                continue
            summary = {k: v for k, v in entry.items() if k != "spec"}
            # local edits win locally, but never lose the remote vote count
            if slug in rows:
                summary["votes"] = max(summary.get("votes", 0), rows[slug].get("votes", 0))
            rows[slug] = summary
    return list(rows.values())


def vote(slug: str, username: str, token: str | None, direction: int = 1) -> StoreResult:
    """One vote per HF user per workflow. Voting needs a token so we can attribute it."""
    slug = slugify(slug)
    if not token or not username:
        return StoreResult(False, "voting needs your Hugging Face token so votes are "
                                  "attributable (Settings → token)")
    api = _api(token)
    try:
        payload = _read_remote_json(_votes_path(slug), token) or {"slug": slug, "voters": {}}
        voters = payload.setdefault("voters", {})
        if direction == 0:
            voters.pop(username, None)
        else:
            voters[username] = 1 if direction > 0 else -1
        payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload["votes"] = sum(1 for v in voters.values() if v > 0)
        api.upload_file(
            path_or_fileobj=json.dumps(payload, indent=2).encode(),
            path_in_repo=_votes_path(slug),
            repo_id=DATASET_ID,
            repo_type="dataset",
            commit_message=f"vote: {slug} {'+' if direction > 0 else '-' if direction < 0 else 'clear'}1 "
                           f"by {username}",
        )
        index = _read_remote_json(INDEX_FILE, token) or {"workflows": {}}
        if slug in (index.get("workflows") or {}):
            index["workflows"][slug]["votes"] = payload["votes"]
            api.upload_file(
                path_or_fileobj=json.dumps(index, indent=2).encode(),
                path_in_repo=INDEX_FILE,
                repo_id=DATASET_ID,
                repo_type="dataset",
                commit_message=f"index: {slug} votes={payload['votes']}",
            )
        return StoreResult(True, f"recorded your vote for '{slug}'", data=payload)
    except Exception as exc:
        return StoreResult(False, f"voting failed: {type(exc).__name__}: {str(exc)[:200]}")


# ─── leaderboard ──────────────────────────────────────────────────────────────


def leaderboard(
    rows: list[dict[str, Any]],
    modality: str = "any",
    industry: str = "any",
    license_filter: str = "commercial-only",
    compute_tier: str = "any",
    sort: str = "votes",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Filter and rank workflows.

    ``license_filter`` defaults to commercial-only because the whole point of the registry
    review is that people can safely ship what they build here; the permissive view exists
    for people who know what they need.
    """
    out = []
    for row in rows:
        bricks = row.get("brick_ids") or []
        if modality != "any" and modality not in (row.get("modalities") or []):
            # fall back to checking the brick ids themselves (mirror entries may lack tags)
            if not any(modality in b for b in bricks):
                continue
        if industry != "any" and row.get("industry") != industry:
            if industry not in (row.get("tags") or []):
                continue
        if license_filter == "commercial-only" and row.get("license_posture") != "commercial-only":
            continue
        if compute_tier != "any" and row.get("compute_tier") != compute_tier:
            continue
        out.append(row)

    keys = {
        "votes": lambda r: (-(r.get("votes") or 0), r.get("name", "")),
        "recent": lambda r: (r.get("published_at") or r.get("updated_at") or "",),
        "steps": lambda r: (r.get("steps") or 0, r.get("name", "")),
        "name": lambda r: (r.get("name", ""),),
    }
    out.sort(key=keys.get(sort, keys["votes"]), reverse=(sort == "recent"))
    return out[:limit]


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Small aggregate for the leaderboard header."""
    from collections import Counter

    by_industry = Counter(r.get("industry", "general") for r in rows)
    by_brick: Counter = Counter()
    for row in rows:
        by_brick.update(row.get("brick_ids") or [])
    return {
        "workflows": len(rows),
        "votes": sum(r.get("votes") or 0 for r in rows),
        "commercial_only": sum(1 for r in rows if r.get("license_posture") == "commercial-only"),
        "by_industry": dict(by_industry.most_common(6)),
        "top_bricks": dict(by_brick.most_common(6)),
        "authors": len({r.get("author", "anonymous") for r in rows}),
    }


def export_spec(spec: WorkflowSpec) -> str:
    return json.dumps(spec.to_dict(), indent=2)


def import_spec(text: str) -> WorkflowSpec | None:
    from daggrstudio.jsonutil import parse_json_object

    payload = parse_json_object(text)
    if not payload:
        return None
    if "spec" in payload and isinstance(payload["spec"], dict):
        payload = payload["spec"]
    if "steps" not in payload:
        return None
    return WorkflowSpec.from_dict(payload)


def cache_age() -> float:
    return time.time()