"""
The Medic: autonomous repair of a workflow, and advice on improving it.

Loop (each round is cheap and bounded):

    validate -> deterministic autofixes -> validate
             -> if still blocking: ask a small model for patch OPS
             -> apply (or reject) the ops -> validate again

Why this shape:
* deterministic fixes first keeps the model out of decisions that are already certain;
* the model only ever emits ops from a closed vocabulary (``patching.OPS``), so a bad
  suggestion is *rejected with a reason* rather than corrupting the workflow;
* the loop stops on no-progress, so it can never spin; every round is recorded in
  ``spec.heal_log`` and surfaced in the UI, so the user can see what changed and why.

The same loop powers the two "it broke later" cases: a sister Space that renamed a
parameter (fixed deterministically by rename) and a sister Space that died or went NC
(fixed by failing over to a verified alternative brick).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from daggrstudio.llm import DEFAULT_MODEL, LLMClient, LLMError, PoolExhausted
from daggrstudio.medic.autofix import AutoFix, run_autofixes
from daggrstudio.medic.patching import OPS, apply_ops, normalise_ops
from daggrstudio.registry.bricks import BrickRegistry, get_registry
from daggrstudio.spec import WorkflowSpec
from daggrstudio.validator import Issue, Report, validate

MAX_ROUNDS = 4

SYSTEM_PROMPT = """You are the Medic for Daggr Studio, a pipeline builder for the daggr library.
A workflow spec failed validation. You repair it by emitting a JSON list of operations.

Hard rules:
- Use ONLY brick ids that appear in the provided catalogue. Never invent a Space or model.
- Endpoint and parameter names must come from the catalogue/observations, never guessed.
- Prefer the smallest change that makes the spec valid.
- If an issue cannot be fixed with the allowed operations, leave it and explain in "notes".
- Reply with JSON only, no prose, no markdown fences.

Allowed operations:
  {"op":"set_input","step":"<id>","param":"<name>","from":"user:<port>|step:<id>.<port>"}
  {"op":"set_input","step":"<id>","param":"<name>","value":<literal>}
  {"op":"drop_input","step":"<id>","param":"<name>"}
  {"op":"set_api_name","step":"<id>","api_name":"/endpoint"}
  {"op":"replace_brick","step":"<id>","brick_id":"<catalogue id>"}
  {"op":"set_outputs","step":"<id>","outputs":{"<port>":{"component":"image","label":"..."}}}
  {"op":"set_postprocess","step":"<id>","hint":"tuple_index:1|dict_path:path|unwrap_path|first"}
  {"op":"add_step","step":{"id":"..","kind":"space|inference|fn","brick_id":"..","api_name":"..",
      "inputs":{...},"outputs":{...},"why":".."}}
  {"op":"remove_step","step":"<id>"}
  {"op":"declare_inputs","inputs":[{"port":"prompt","component":"textbox","label":"...","default":"..."}]}
  {"op":"set_meta","name":"..","industry":"..","tags":[".."],"license_posture":"commercial-only|any","compute_tier":"cloud-free|cloud-paid","notes":".."}
  {"op":"set_concurrent","step":"<id>","concurrent":true}

Response format:
{"diagnosis":"one sentence","ops":[...],"notes":"anything you could not fix"}"""


@dataclass
class HealEvent:
    round: int
    actor: str               # deterministic | model
    summary: str
    codes_before: list[str] = field(default_factory=list)
    codes_after: list[str] = field(default_factory=list)
    model: str = ""
    applied: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    fixes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "actor": self.actor,
            "summary": self.summary,
            "codes_before": self.codes_before,
            "codes_after": self.codes_after,
            "model": self.model,
            "applied": self.applied,
            "skipped": self.skipped,
            "fixes": self.fixes,
        }


@dataclass
class HealResult:
    spec: WorkflowSpec
    report: Report
    log: list[HealEvent] = field(default_factory=list)
    status: str = "unknown"   # healthy | healed | needs_human | model_unavailable
    model: str = ""
    note: str = ""

    @property
    def healed(self) -> bool:
        return self.status in ("healthy", "healed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "note": self.note,
            "report": self.report.to_dict(),
            "log": [e.to_dict() for e in self.log],
            "spec": self.spec.to_dict(),
        }

    def timeline(self) -> list[str]:
        out = []
        for event in self.log:
            icon = "🔧" if event.actor == "deterministic" else "🩺"
            out.append(f"{icon} round {event.round} ({event.actor}): {event.summary}")
        return out


def _issue_brief(issues: list[Issue]) -> str:
    lines = []
    for issue in issues:
        where = f" step={issue.step_id}" if issue.step_id else ""
        port = f" port={issue.port}" if issue.port else ""
        lines.append(f"- [{issue.code}]{where}{port} {issue.message}"
                     + (f" (hint: {issue.fix_hint})" if issue.fix_hint else ""))
    return "\n".join(lines)


def _catalogue_for(spec: WorkflowSpec, registry: BrickRegistry, report: Report) -> str:
    """Only show the model the bricks it might plausibly need: smaller prompt, better choice."""
    from daggrstudio.codegen.fn_library import fn_catalogue_text

    modalities = set()
    for step in spec.steps:
        brick = registry.by_id(step.brick_id)
        if brick:
            modalities.add(brick.modality)
    broken = {i.data.get("modality") for i in report.issues if i.data.get("modality")}
    modalities |= {m for m in broken if m}
    if not modalities:
        modalities = set(registry.modalities_present())
    wanted = set(modalities) | {"image-gen", "image-edit", "utility"}
    bricks = registry.find(modalities=sorted(wanted),
                           commercial_only=spec.license_posture == "commercial-only",
                           include_not_running=False)
    text = registry.catalogue_text(bricks, max_bricks=32)
    return (
        "BRICK CATALOGUE (authoritative; only these ids exist)\n" + text +
        "\n\nLOCAL HELPERS for kind=fn steps (only these exist)\n" + fn_catalogue_text()
    )


def _repair_prompt(spec: WorkflowSpec, registry: BrickRegistry, report: Report,
                   skipped: list[dict[str, Any]]) -> str:
    import json

    parts = [
        "GOAL: " + (spec.intent or spec.name),
        f"LICENCE POSTURE: {spec.license_posture} | COMPUTE TIER: {spec.compute_tier}",
        "",
        "CURRENT SPEC (JSON):",
        json.dumps(spec.to_dict(), indent=1)[:6000],
        "",
        "BLOCKING ISSUES:",
        _issue_brief(report.blocking) or "(none)",
    ]
    if report.warnings:
        parts += ["", "WARNINGS (fix if cheap):", _issue_brief(report.warnings[:8])]
    if skipped:
        parts += ["", "YOUR PREVIOUS OPS THAT WERE REJECTED (do not repeat them):",
                  json.dumps(skipped, indent=1)[:1200]]
    parts += ["", _catalogue_for(spec, registry, report), "",
              "Emit the JSON repair now."]
    return "\n".join(parts)


def heal(
    spec: WorkflowSpec,
    registry: BrickRegistry | None = None,
    client: LLMClient | None = None,
    token: str | None = None,
    max_rounds: int = MAX_ROUNDS,
    live: bool = True,
    deterministic: bool = True,
    model: str | None = None,
) -> HealResult:
    """
    Validate and repair ``spec`` (a copy; the caller's spec is untouched).

    Set ``client=None`` for a purely deterministic pass (no token needed): that alone fixes
    dead bricks, licence problems and renamed parameters.
    """
    registry = registry or get_registry()
    spec = spec.clone()
    result = HealResult(spec=spec, report=validate(spec, registry, live=live, token=token),
                        model=model or (client.model if client else ""))

    if result.report.ok:
        result.status = "healthy"
        return result

    seen_codes: set[tuple[str, ...]] = set()
    skipped_ops: list[dict[str, Any]] = []
    stall_rounds = 0

    for round_no in range(1, max_rounds + 1):
        before = result.report.codes()

        if deterministic and round_no == 1:
            fixes = run_autofixes(spec, result.report, registry)
            after_report = validate(spec, registry, live=live, token=token)
            if fixes:
                result.log.append(HealEvent(
                    round=round_no, actor="deterministic",
                    summary="; ".join(f.detail for f in fixes[:4]),
                    codes_before=before, codes_after=after_report.codes(),
                    fixes=[f.to_dict() for f in fixes],
                ))
                result.report = after_report
                if result.report.ok:
                    result.status = "healed"
                    break
                continue

        if client is None:
            result.status = "needs_human"
            result.note = "deterministic repairs exhausted; no model configured for the next round"
            break

        prompt = _repair_prompt(spec, registry, result.report, skipped_ops)
        try:
            payload, used_model = client.ask_json(prompt, system=SYSTEM_PROMPT,
                                                  model=model or client.model)
        except (LLMError, PoolExhausted) as exc:
            result.status = "model_unavailable"
            result.note = str(exc)[:300]
            break

        ops = normalise_ops(payload)
        if not ops:
            result.status = "needs_human"
            result.note = "the model returned no usable operations"
            break

        patch = apply_ops(spec, ops, registry)
        after_report = validate(spec, registry, live=live, token=token)
        result.log.append(HealEvent(
            round=round_no, actor="model",
            summary=(payload or {}).get("diagnosis", "") or patch.summary(),
            codes_before=before, codes_after=after_report.codes(),
            model=used_model, applied=patch.applied, skipped=patch.skipped,
        ))
        result.model = used_model
        result.report = after_report
        skipped_ops = patch.skipped

        if result.report.ok:
            result.status = "healed"
            break
        if not patch.changed:
            # Nothing applied. Give the model exactly one round to react to the rejection
            # reasons (that feedback is the most useful signal we can hand it), then stop.
            if patch.skipped and stall_rounds < 1:
                stall_rounds += 1
                continue
            result.status = "needs_human"
            result.note = "no operation was applicable; the workflow needs a human decision"
            break
        signature = tuple(result.report.codes())
        if signature in seen_codes:
            result.status = "needs_human"
            result.note = "repair rounds stopped converging"
            break
        seen_codes.add(signature)
    else:
        result.status = "needs_human"
        result.note = f"still blocking after {max_rounds} rounds"

    spec.heal_log = [e.to_dict() for e in result.log]
    return result


# ─── upgrade advisor ──────────────────────────────────────────────────────────


ADVISOR_SYSTEM = """You advise on upgrading bricks in a daggr pipeline.
You are given the current steps and candidate replacement bricks. Recommend swaps only when
the candidate is a genuine improvement for the stated goal (quality, speed, licence, freshness).
Never invent brick ids. Reply with JSON only:
{"suggestions":[{"step":"<id>","brick_id":"<catalogue id>","why":"one sentence","confidence":0.0-1.0}]}
Return {"suggestions":[]} if nothing is worth changing."""


def upgrade_advisor(
    spec: WorkflowSpec,
    registry: BrickRegistry | None = None,
    client: LLMClient | None = None,
) -> list[dict[str, Any]]:
    """
    Suggest newer/better bricks for the current workflow.

    Deterministic fallback (no model): candidates that are live-verified, commercially
    compatible with the workflow's posture, in the same modality, and better liked on the
    Hub than the current brick.
    """
    import json

    registry = registry or get_registry()
    commercial_only = spec.license_posture == "commercial-only"
    current = {s.id: registry.by_id(s.brick_id) for s in spec.steps}
    catalogue: list[Any] = []
    for step in spec.steps:
        brick = current.get(step.id)
        if brick is None:
            continue
        alternatives = [
            b for b in registry.alternatives(brick, commercial_only=commercial_only)
            if b.status == "running" and b.output_kind == brick.output_kind
        ][:6]
        for alt in alternatives:
            catalogue.append((step.id, brick, alt))

    if not catalogue:
        return []

    if client is None:
        suggestions = []
        for step_id, brick, alt in catalogue:
            if (alt.likes or 0) > (brick.likes or 0):
                suggestions.append({
                    "step": step_id,
                    "brick_id": alt.id,
                    "why": f"{alt.source} is more popular on the Hub "
                           f"({alt.likes} vs {brick.likes} likes) and serves the same {brick.modality} role",
                    "confidence": 0.4,
                    "source": "heuristic",
                })
        return suggestions[:5]

    lines = [f"GOAL: {spec.intent or spec.name}", "", "CURRENT STEPS:"]
    for step in spec.steps:
        brick = current.get(step.id)
        lines.append(f"- step '{step.id}' ({(step.title or step.id)}): "
                     f"{brick.id if brick else 'unverified'} "
                     f"({brick.source if brick else '?'}, license {brick.license if brick else '?'})")
    lines.append("")
    lines.append("CANDIDATES:")
    for step_id, brick, alt in catalogue:
        lines.append(f"- replaces step '{step_id}': {alt.id} | {alt.source} | {alt.modality} | "
                     f"license {alt.license} commercial={alt.commercial} | likes={alt.likes} | {alt.notes}")
    lines.append("")
    lines.append("Reply with the JSON now.")

    try:
        payload, _model = client.ask_json("\n".join(lines), system=ADVISOR_SYSTEM)
    except (LLMError, PoolExhausted):
        return []

    out: list[dict[str, Any]] = []
    allowed = {(step_id, alt.id) for step_id, _b, alt in catalogue}
    for item in (payload or {}).get("suggestions", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("step", ""))
        brick_id = str(item.get("brick_id", ""))
        if (step_id, brick_id) not in allowed:
            continue  # hallucinated pairing: drop it silently, the catalogue is the truth
        out.append({
            "step": step_id,
            "brick_id": brick_id,
            "why": str(item.get("why", ""))[:300],
            "confidence": float(item.get("confidence", 0.5) or 0.5),
            "source": "advisor",
        })
    return out[:5]


def apply_suggestions(spec: WorkflowSpec, suggestions: list[dict[str, Any]],
                      registry: BrickRegistry | None = None) -> tuple[WorkflowSpec, list[dict[str, Any]]]:
    """Apply advisor suggestions as replace_brick ops; returns (new spec, applied ops)."""
    registry = registry or get_registry()
    ops = [{"op": "replace_brick", "step": s["step"], "brick_id": s["brick_id"]}
           for s in suggestions]
    out = spec.clone()
    result = apply_ops(out, ops, registry)
    return out, result.applied


__all__ = [
    "AutoFix",
    "HealEvent",
    "HealResult",
    "OPS",
    "apply_suggestions",
    "heal",
    "upgrade_advisor",
]