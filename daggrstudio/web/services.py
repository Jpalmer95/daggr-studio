"""
Service layer behind the Builder UI.

Every user action is a plain function here that takes/returns JSON-able data. The Gradio
layer is a thin shell on top, which keeps the interesting logic (plan -> heal -> run ->
publish) testable offline and reusable from the agent API without touching the UI.
"""

from __future__ import annotations

import json
from typing import Any

from daggrstudio.codegen import render_app_py
from daggrstudio.hub import (
    DATASET_ID,
    leaderboard,
    list_workflows,
    publish,
    spec_from_entry,
    stats,
    vote,
)
from daggrstudio.introspect import introspect
from daggrstudio.llm import LLMClient, list_router_models, pool_status, verify_token
from daggrstudio.medic import apply_suggestions, heal, upgrade_advisor
from daggrstudio.planner import plan
from daggrstudio.registry.bricks import get_registry
from daggrstudio.runner import run_spec
from daggrstudio.spec import WorkflowSpec
from daggrstudio.validator import validate

STARTERS = {
    "Game sprite (cutout + report)": (
        "turn a one-line character idea into a game sprite with a transparent background"
    ),
    "Game character sheet -> 3D": (
        "make a 3D model of a game character from a text description"
    ),
    "Social post pack": (
        "write a social media caption and generate a matching image for a coffee shop"
    ),
    "Narrated video idea": (
        "write a short script and narrate it as a voiceover"
    ),
    "Album art from lyrics": (
        "generate album artwork from a short lyric idea"
    ),
    "Product photo clean-up": (
        "remove the background from a product photo"
    ),
}


# ─── helpers ──────────────────────────────────────────────────────────────────


def as_spec(payload: Any) -> WorkflowSpec | None:
    if isinstance(payload, WorkflowSpec):
        return payload
    if isinstance(payload, dict) and "steps" in payload:
        return WorkflowSpec.from_dict(payload)
    return None


def issues_table(report: Any) -> list[list[str]]:
    rows = []
    for issue in report.issues:
        rows.append([
            {"blocking": "⛔", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "•"),
            issue.code,
            issue.step_id or "",
            (issue.message or "")[:220],
            (issue.fix_hint or "")[:160],
        ])
    return rows


ISSUE_HEADERS = ["", "code", "step", "what", "suggested fix"]


def spec_summary(spec: WorkflowSpec) -> str:
    registry = get_registry()
    lines = [f"### {spec.name}",
             f"*{spec.intent or 'no goal recorded'}*", "",
             f"**{len(spec.steps)} steps** · industry `{spec.industry}` · "
             f"licence posture `{spec.license_posture}` · compute `{spec.compute_tier}` · "
             f"planner `{spec.planner_model or 'hand-built'}`", ""]
    for i, step in enumerate(spec.steps, 1):
        brick = registry.by_id(step.brick_id)
        badge = ""
        if brick:
            badge = (f" · `{brick.license}`"
                     f"{' · commercial ✓' if brick.commercial else ' · ⚠ non-commercial'}"
                     f" · {brick.status}")
        label = brick.source if brick else (step.fn or step.source or "?")
        lines.append(f"{i}. **{step.title or step.id}** — {label}{badge}")
        if step.why:
            lines.append(f"   - {step.why}")
    if spec.inputs:
        lines += ["", "Inputs: " + ", ".join(f"`{p.port}` ({p.component})" for p in spec.inputs)]
    return "\n".join(lines)


def heal_timeline_md(result: Any) -> str:
    if not result.log:
        return "No repairs were needed." if result.healed else "No repairs were attempted."
    lines = []
    for event in result.log:
        icon = "🔧" if event.actor == "deterministic" else "🩺"
        lines.append(f"{icon} **round {event.round}** ({event.actor}) — {event.summary}")
        if event.codes_before:
            lines.append(f"   - before: `{', '.join(event.codes_before)}`")
        if event.codes_after:
            lines.append(f"   - after: `{', '.join(event.codes_after)}`")
        for fix in event.fixes:
            lines.append(f"   - fixed: {fix.get('detail', '')}")
        for skipped in event.skipped[:3]:
            lines.append(f"   - rejected: `{skipped.get('op')}` — {skipped.get('reason')}")
    return "\n".join(lines)


# ─── planning ─────────────────────────────────────────────────────────────────


def plan_workflow(
    intent: str,
    token: str | None = None,
    model: str | None = None,
    industry: str = "general",
    license_posture: str = "commercial-only",
    max_steps: int = 5,
    live_validation: bool = True,
    heal_rounds: int = 4,
) -> dict[str, Any]:
    """
    Intent -> healed, validated workflow. This is the whole product in one function.

    Returns a payload the UI can render directly:
    ``{ok, status, message, spec, code, issues, timeline, model, notes}``.
    """
    registry = get_registry()
    out: dict[str, Any] = {"ok": False, "status": "failed", "message": "",
                           "spec": None, "code": "", "issues": [],
                           "timeline": "", "model": "", "notes": [], "rows": []}

    if not intent or not intent.strip():
        out["message"] = "Describe what you want to build first."
        return out

    client = LLMClient(token=token, model=model) if (token or model) else None
    if client is None:
        # fall back to the space's own token (metered) - resolve_token handles precedence
        client = LLMClient(token=None, model=model)

    planned = plan(intent, registry=registry, client=client, industry=industry,
                   license_posture=license_posture, max_steps=max_steps, model=model)
    out["notes"] = planned.notes
    out["model"] = planned.model
    if not planned.ok:
        out["message"] = planned.error or "planning failed"
        return out

    spec = planned.spec
    assert spec is not None

    healed = heal(spec, registry=registry, client=client, token=token,
                  max_rounds=heal_rounds, live=live_validation, model=model)
    spec = healed.spec
    out.update({
        "spec": spec.to_dict(),
        "code": render_app_py(spec, registry),
        "status": healed.status,
        "model": healed.model or out["model"],
        "timeline": heal_timeline_md(healed),
        "issues": issues_table(healed.report),
        "ok": healed.healed,
        "message": ("ready to run" if healed.healed
                    else f"needs attention: {healed.note or healed.report.summary()}"),
        "summary": spec_summary(spec),
        "pool": pool_status(),
    })
    out["canvas"] = (publish_to_canvas(spec) if healed.healed
                     else "not pushed: the workflow still has blocking findings")
    return out


def validate_spec(payload: Any, live: bool = True, token: str | None = None) -> dict[str, Any]:
    spec = as_spec(payload)
    if spec is None:
        return {"ok": False, "message": "no workflow yet"}
    report = validate(spec, get_registry(), live=live, token=token)
    return {"ok": report.ok, "issues": issues_table(report), "message": report.summary(),
            "codes": report.codes()}


def heal_spec(payload: Any, token: str | None = None, model: str | None = None,
              live: bool = True) -> dict[str, Any]:
    spec = as_spec(payload)
    if spec is None:
        return {"ok": False, "message": "no workflow yet"}
    client = LLMClient(token=token, model=model)
    result = heal(spec, registry=get_registry(), client=client, token=token, live=live,
                  model=model)
    out = {
        "ok": result.healed,
        "status": result.status,
        "spec": result.spec.to_dict(),
        "code": render_app_py(result.spec, get_registry()),
        "issues": issues_table(result.report),
        "timeline": heal_timeline_md(result),
        "summary": spec_summary(result.spec),
        "message": result.note or result.report.summary(),
        "model": result.model,
    }
    out["canvas"] = (publish_to_canvas(result.spec) if result.healed
                     else "not pushed: still blocking after healing")
    return out


def render_code(payload: Any) -> str:
    spec = as_spec(payload)
    return render_app_py(spec, get_registry()) if spec else ""


# ─── running ──────────────────────────────────────────────────────────────────


def run_workflow(payload: Any, values: dict[str, Any] | None = None,
                 token: str | None = None) -> dict[str, Any]:
    spec = as_spec(payload)
    if spec is None:
        return {"ok": False, "message": "no workflow yet", "rows": [], "artifacts": []}
    result = run_spec(spec, values=values or {}, token=token, registry=get_registry())
    rows = []
    for step in result.steps:
        rows.append([
            {"ok": "✅", "repaired": "🔧", "failed": "⛔"}.get(step.status, "•"),
            step.title or step.step_id,
            f"{step.seconds:.1f}s",
            step.brick_id or "-",
            " · ".join(step.repairs)[:200] or (step.error[:200] if step.error else ""),
        ])
    return {
        "ok": result.ok,
        "message": result.summary() + (f" — {result.note}" if result.note else ""),
        "rows": rows,
        "artifacts": result.artifacts,
        "repaired_spec": result.repaired_spec.to_dict() if result.repaired_spec else None,
        "raw": result.to_dict(),
    }


RUN_HEADERS = ["", "step", "time", "brick", "repairs / error"]


# ─── sharing ──────────────────────────────────────────────────────────────────


def publish_workflow(payload: Any, token: str | None = None, note: str = "") -> dict[str, Any]:
    spec = as_spec(payload)
    if spec is None:
        return {"ok": False, "message": "no workflow yet"}
    username = verify_token(token) if token else ""
    result = publish(spec, token=token, username=username or "", note=note)
    return {"ok": result.ok, "message": result.message, "url": result.url,
            "slug": spec.slug}


def load_workflow(slug: str, token: str | None = None) -> dict[str, Any]:
    from daggrstudio.hub import load

    result = load((slug or "").strip(), token=token)
    if not result.ok:
        return {"ok": False, "message": result.message}
    spec = spec_from_entry(result.data)
    if spec is None:
        return {"ok": False, "message": "that entry does not contain a usable workflow"}
    return {"ok": True, "message": result.message, "spec": spec.to_dict(),
            "code": render_app_py(spec, get_registry()), "summary": spec_summary(spec),
            "canvas": publish_to_canvas(spec)}


def leaderboard_view(modality: str = "any", industry: str = "any",
                     license_filter: str = "commercial-only", compute_tier: str = "any",
                     sort: str = "votes", token: str | None = None) -> dict[str, Any]:
    rows = list_workflows(token=token)
    filtered = leaderboard(rows, modality=modality, industry=industry,
                           license_filter=license_filter, compute_tier=compute_tier,
                           sort=sort)
    table = [[r.get("name", ""), r.get("author", ""), r.get("industry", ""),
              ", ".join(r.get("brick_ids") or [])[:70], r.get("steps", ""),
              r.get("votes", 0), r.get("license_posture", ""),
              (r.get("published_at") or "")[:10]] for r in filtered]
    summary = stats(filtered)
    md = (f"**{summary['workflows']} workflows** · {summary['votes']} votes · "
          f"{summary['authors']} contributors · "
          f"{summary['commercial_only']} commercial-only\n\n"
          f"Most used bricks: " +
          ", ".join(f"`{k}`×{v}" for k, v in summary["top_bricks"].items()))
    return {"ok": True, "rows": table, "message": md, "stats": summary,
            "dataset": DATASET_ID, "slugs": [r.get("slug") for r in filtered]}


LEADERBOARD_HEADERS = ["workflow", "author", "industry", "bricks", "steps", "votes",
                       "licence", "published"]


def cast_vote(slug: str, direction: int, token: str | None = None) -> dict[str, Any]:
    username = verify_token(token) if token else ""
    result = vote(slug, username=username or "", token=token, direction=direction)
    return {"ok": result.ok, "message": result.message}


# ─── bricks ───────────────────────────────────────────────────────────────────


def bricks_table(modality: str = "any", only_commercial: bool = True,
                 only_running: bool = False, limit: int = 60) -> list[list[str]]:
    registry = get_registry()
    bricks = registry.find(
        modalities=None if modality == "any" else [modality],
        commercial_only=only_commercial,
        include_not_running=not only_running,
        limit=limit,
    )
    rows = []
    for brick in bricks:
        rows.append([
            brick.id,
            f"{brick.source}{brick.api_name or ''}",
            brick.modality,
            brick.output_kind,
            brick.license,
            "✓" if brick.commercial else "—",
            brick.status,
            str(brick.probe) if hasattr(brick, "probe") else "",
            ", ".join(brick.industries[:3]),
            (brick.notes or "")[:80],
        ])
    return rows


BRICK_HEADERS = ["id", "source", "modality", "outputs", "licence", "commercial", "status",
                 "probe", "industries", "notes"]


def check_workflow_bricks(payload: Any, token: str | None = None, probe: bool = True) -> dict[str, Any]:
    """
    Re-check the bricks this workflow uses, right now.

    This is the in-app version of "a sister Space changed under me": we re-read each Space's
    live API, optionally make one real call, and report what moved. Then ``heal_spec`` can
    repair the workflow against the new reality.
    """
    spec = as_spec(payload)
    if spec is None:
        return {"ok": False, "message": "no workflow yet", "rows": []}
    registry = get_registry()
    rows: list[list[str]] = []
    changes: list[str] = []
    for step in spec.steps:
        brick = registry.by_id(step.brick_id)
        if brick is None or step.kind == "fn":
            continue
        info = introspect(brick.source, brick.api_name, force=True, token=token)
        if not info.ok:
            rows.append(["⛔", brick.source, brick.api_name or "-", "unreachable",
                         (info.error or "")[:120]])
            changes.append(f"`{brick.source}` is unreachable: {info.error}")
            continue
        live = info.param_names
        missing = [p for p in step.inputs if p not in live]
        if missing:
            changes.append(f"`{brick.source}` no longer accepts {missing}; "
                           f"live params: {live[:8]}")
        rows.append(["✅" if not missing else "⚠️", brick.source, brick.api_name or "-",
                     f"{len(live)} params", ", ".join(live[:10])])
    message = "Everything the workflow depends on still looks the same." if not changes else \
        "Changes detected:\n" + "\n".join(f"- {c}" for c in changes)
    return {"ok": not changes, "rows": rows, "message": message,
            "headers": ["", "space", "endpoint", "state", "live parameters"]}


def advise_upgrades(payload: Any, token: str | None = None, model: str | None = None) -> dict[str, Any]:
    spec = as_spec(payload)
    if spec is None:
        return {"ok": False, "message": "no workflow yet", "suggestions": []}
    client = LLMClient(token=token, model=model)
    suggestions = upgrade_advisor(spec, registry=get_registry(), client=client)
    if not suggestions:
        return {"ok": True, "message": "No upgrades worth making right now.",
                "suggestions": []}
    # validate the suggestion set: never offer something that breaks the licence posture
    registry = get_registry()
    safe = []
    for suggestion in suggestions:
        brick = registry.by_id(suggestion["brick_id"])
        if brick is None:
            continue
        if spec.license_posture == "commercial-only" and not brick.commercial:
            continue
        safe.append(suggestion)
    md = "\n".join(f"- **{s['step']}** → `{s['brick_id']}` ({s.get('why', '')})"
                   for s in safe) or "Nothing safe to suggest."
    return {"ok": True, "message": md, "suggestions": safe}


def apply_upgrade(payload: Any, brick_id: str = "", step_id: str = "",
                  token: str | None = None) -> dict[str, Any]:
    spec = as_spec(payload)
    if spec is None or not brick_id or not step_id:
        return {"ok": False, "message": "pick a workflow and an upgrade first"}
    updated, applied = apply_suggestions(
        spec, [{"step": step_id, "brick_id": brick_id}], registry=get_registry())
    if not applied:
        return {"ok": False, "message": "that upgrade was rejected (unknown brick or step)"}
    return {"ok": True, "message": f"switched `{step_id}` to `{brick_id}`",
            "spec": updated.to_dict(), "code": render_app_py(updated, get_registry()),
            "summary": spec_summary(updated), "canvas": publish_to_canvas(updated)}


# ─── settings / info ──────────────────────────────────────────────────────────


def settings_info(token: str | None = None) -> dict[str, Any]:
    registry = get_registry()
    username = verify_token(token) if token else ""
    return {
        "username": username or "",
        "pool": pool_status(),
        "registry": registry.summary(),
        "models": list_router_models(token),
        "dataset": DATASET_ID,
        "token_ok": bool(username),
        "token_source": "yours" if username else ("space pool" if pool_status()["remaining"] else "none"),
    }


# ─── deploy the workflow as its own Space ─────────────────────────────────────


def deploy_space(payload: Any, token: str | None, space_name: str = "") -> dict[str, Any]:
    """
    One click: give the user their own Space running their workflow on daggr's real canvas.

    BYOK by design - the Space is created under *their* account with *their* token, so we
    never host someone else's workflow at our cost.
    """
    spec = as_spec(payload)
    if spec is None:
        return {"ok": False, "message": "no workflow yet"}
    if not token:
        return {"ok": False, "message": "deploying needs your Hugging Face token "
                                        "(Settings → token)"}
    username = verify_token(token)
    if not username:
        return {"ok": False, "message": "that token was rejected by Hugging Face"}

    slug = (space_name or f"daggr-{spec.slug}").strip().lower().replace("_", "-")[:48]
    repo_id = f"{username}/{slug}"
    code = render_app_py(spec, get_registry())

    readme = f"""---
title: {spec.name[:60]}
emoji: 🧱
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "5.24.0"
app_file: app.py
pinned: false
short_description: daggr workflow built with Daggr Studio
---

# {spec.name}

Built with [Daggr Studio](https://huggingface.co/spaces/jkorstad/daggr-studio) from verified bricks.

{spec.intent}

## Bricks used

{chr(10).join('- `' + b + '`' for b in spec.brick_ids) or '- (none)'}

Licence posture: `{spec.license_posture}`. Run locally with `pip install daggr && daggr app.py`.
"""
    requirements = "daggr==0.8.0\nhuggingface_hub\n"

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_file(path_or_fileobj=code.encode(), path_in_repo="app.py",
                        repo_id=repo_id, repo_type="space",
                        commit_message="feat: workflow from Daggr Studio")
        api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="space",
                        commit_message="docs: readme")
        api.upload_file(path_or_fileobj=requirements.encode(), path_in_repo="requirements.txt",
                        repo_id=repo_id, repo_type="space",
                        commit_message="chore: requirements")
        if token:
            try:
                api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
            except Exception:
                pass
        return {"ok": True, "url": f"https://huggingface.co/spaces/{repo_id}",
                "message": f"created https://huggingface.co/spaces/{repo_id} — it will be "
                           f"live in a minute or two (build logs there if it misbehaves)"}
    except Exception as exc:
        return {"ok": False, "message": f"deploy failed: {type(exc).__name__}: {str(exc)[:200]}"}



def publish_to_canvas(spec: Any) -> str:
    """
    Push a workflow to the daggr canvas (the visual view at /canvas/).

    Lives here rather than in any one caller so the REST API, the queued Gradio endpoints and
    the legacy Builder all agree on what the canvas is showing. A canvas failure is reported,
    never raised: it must not be able to break the thing that actually builds workflows.
    """
    try:
        from daggrstudio.web.canvas import show_spec

        parsed = as_spec(spec)
        return show_spec(parsed) if parsed is not None else "nothing to show"
    except Exception as exc:
        return f"canvas unavailable: {type(exc).__name__}: {exc}"


def export_spec_json(payload: Any) -> str:
    spec = as_spec(payload)
    return json.dumps(spec.to_dict(), indent=2) if spec else ""


def import_payload(text: str) -> Any:
    """Accept a bare spec dict or a published entry ({"spec": {...}})."""
    from daggrstudio.hub import import_spec

    spec = import_spec(text or "")
    return spec.to_dict() if spec else None


def list_models_for_ui(token: str | None = None) -> list[str]:
    """Model dropdown choices: the curated shortlist first, then everything on the router."""
    from daggrstudio.llm import SUGGESTED_MODELS, list_router_models

    live = []
    try:
        live = list_router_models(token)
    except Exception:
        live = []
    ordered = list(SUGGESTED_MODELS)
    for model in live:
        if model not in ordered:
            ordered.append(model)
    return ordered[:200]


def default_model() -> str:
    from daggrstudio.llm import DEFAULT_MODEL

    return DEFAULT_MODEL


def pool_snapshot() -> dict[str, Any]:
    """Community-pool state, safe to show in the UI (never contains a token)."""
    from daggrstudio.llm import pool_status

    try:
        return pool_status()
    except Exception:
        return {"cap": 0, "used": 0, "remaining": 0, "date": ""}


def verify_username(token: str | None) -> str | None:
    """HF username for a token, or None when it is missing/invalid."""
    if not token:
        return None
    from daggrstudio.llm import verify_token

    return verify_token(token)