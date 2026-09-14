"""
Deterministic repairs.

Order matters in the Medic: fix what can be fixed with certainty *before* spending a model
call. These three cases cover most real-world breakage, and they are decided by the registry
and by string similarity, not by a model's opinion:

1. ``PARAM_RENAMED`` / ``PARAM_MISSING``  -> remap when the live Space has an obvious match
2. ``BRICK_BROKEN`` / ``SPACE_UNREACHABLE`` -> fail over to a compatible alternative brick
3. ``LICENSE_NON_COMMERCIAL`` -> swap to a commercially licensed equivalent

Anything these cannot settle deterministically is handed to the LLM with the remaining
findings - and, crucially, the reason each automatic fix was taken is recorded so the user
can audit what their workflow became.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from daggrstudio.registry.bricks import Brick, BrickRegistry
from daggrstudio.spec import Binding, Step, WorkflowSpec
from daggrstudio.validator import Issue, Report, similarity_suggestions

#: A rename is only applied automatically when the match is this close.
RENAME_CUTOFF = 0.62


@dataclass
class AutoFix:
    kind: str
    step_id: str
    detail: str
    before: str = ""
    after: str = ""
    issue_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "step_id": self.step_id,
            "detail": self.detail,
            "before": self.before,
            "after": self.after,
            "issue_codes": self.issue_codes,
        }


def _rename_binding(step: Step, old: str, new: str) -> None:
    """Move a binding to a new param name, preserving its wiring."""
    if old not in step.inputs or new in step.inputs:
        return
    step.inputs[new] = step.inputs.pop(old)


def autofix_renamed_params(
    spec: WorkflowSpec,
    report: Report,
    registry: BrickRegistry,
) -> list[AutoFix]:
    """
    Handle ``PARAM_RENAMED`` by remapping to the live parameter when it is unambiguous.

    This is the "a sister Space updated and broke my workflow" repair: the Space keeps
    working, someone renames `prompt` to `text`, and the workflow heals itself.
    """
    fixes: list[AutoFix] = []
    for issue in report.issues:
        if issue.code != "PARAM_RENAMED" or not issue.step_id or not issue.port:
            continue
        step = spec.step(issue.step_id)
        if step is None or issue.port not in step.inputs:
            continue
        live = list(issue.data.get("live_params") or [])
        suggestions = list(issue.data.get("suggestions") or [])
        if not suggestions:
            suggestions = similarity_suggestions(issue.port, live)
        if not suggestions:
            continue

        # a suggestion that collides with an already-wired param is not a safe rename
        candidate = next((s for s in suggestions if s not in step.inputs), None)
        if candidate is None:
            continue

        # additionally require the keyword pair to look like the same concept
        binding = step.inputs[issue.port]
        if binding.kind == "step":
            # let the model decide: silently repointing an edge is riskier than a rename
            continue

        _rename_binding(step, issue.port, candidate)
        fixes.append(AutoFix(
            kind="param_rename",
            step_id=step.id,
            detail=f"'{issue.port}' no longer exists on the live Space; rewired to "
                   f"'{candidate}' (closest live parameter)",
            before=issue.port,
            after=candidate,
            issue_codes=[issue.code],
        ))
    return fixes


def autofix_missing_params(
    spec: WorkflowSpec,
    report: Report,
    registry: BrickRegistry,
) -> list[AutoFix]:
    """
    Handle ``PARAM_MISSING`` when the missing param is clearly the step's main input and a
    sibling param can power it (e.g. a Space that needs both `image` and `img`).
    """
    fixes: list[AutoFix] = []
    for issue in report.issues:
        if issue.code != "PARAM_MISSING" or not issue.step_id or not issue.port:
            continue
        step = spec.step(issue.step_id)
        if step is None or issue.port in step.inputs:
            continue
        live = list(issue.data.get("live_params") or [])
        # find an existing binding that is a file/image edge and a live param of the same ilk
        for param, binding in list(step.inputs.items()):
            if not (param.lower() in ("image", "img", "photo", "input_image", "init_image")
                    and binding.kind == "step"):
                continue
            if issue.port.lower() not in ("image", "img", "photo", "input_image", "init_image"):
                continue
            if issue.port in live:
                step.inputs[issue.port] = Binding(source=binding.source)
                fixes.append(AutoFix(
                    kind="param_alias",
                    step_id=step.id,
                    detail=f"Space also expects '{issue.port}'; fed it the same image edge "
                           f"as '{param}'",
                    before="",
                    after=issue.port,
                    issue_codes=[issue.code],
                ))
            break
    return fixes


def _compatible(brick: Brick, step: Step, registry: BrickRegistry) -> bool:
    """An alternative must keep the step's role: same output kind, at least one file param
    when the step consumes a file."""
    if brick.output_kind != (registry.by_id(step.brick_id).output_kind
                             if registry.by_id(step.brick_id) else brick.output_kind):
        return False
    if brick.kind == "space" and not brick.api_name:
        return False
    return True


def autofix_broken_bricks(
    spec: WorkflowSpec,
    report: Report,
    registry: BrickRegistry,
    commercial_only: bool | None = None,
) -> list[AutoFix]:
    """
    Fail over to an alternative brick when the chosen one is dead.

    Applies to ``BRICK_BROKEN`` (verification failed) and ``SPACE_UNREACHABLE`` (the live
    Space would not answer). The replacement is recorded in the heal log so the user always
    knows their workflow now runs a different model.
    """
    fixes: list[AutoFix] = []
    commercial_only = (spec.license_posture == "commercial-only"
                       if commercial_only is None else commercial_only)

    for issue in report.issues:
        if issue.code not in ("BRICK_BROKEN", "SPACE_UNREACHABLE", "API_NAME_UNVERIFIED"):
            continue
        step = spec.step(issue.step_id or "")
        if step is None:
            continue
        current = registry.by_id(step.brick_id)
        candidates = registry.alternatives(current, commercial_only=commercial_only) if current \
            else registry.find(modalities=[issue.data.get("modality") or "utility"],
                              commercial_only=commercial_only)
        # only accept a candidate that is verified live and shares the output kind
        target = None
        for cand in candidates:
            if not cand.is_live_verified and cand.status != "running":
                continue
            if current and cand.output_kind != current.output_kind:
                continue
            if cand.kind == "space" and not cand.api_name:
                continue
            target = cand
            break
        if target is None:
            continue

        old_source = current.source if current else (step.source or "")
        step.brick_id = target.id
        step.source = target.source
        step.api_name = target.api_name
        step.kind = "inference" if target.kind == "inference_model" else "space"
        # rewire params: keep bindings whose names still exist, alias obvious renames
        _rewire_after_swap(step, target)
        fixes.append(AutoFix(
            kind="brick_failover",
            step_id=step.id,
            detail=f"{old_source} was unusable ({issue.code}); switched to "
                   f"{target.source} ({target.id}, license {target.license})",
            before=old_source,
            after=target.source,
            issue_codes=[issue.code],
        ))
    return fixes


def _rewire_after_swap(step: Step, brick: Brick) -> None:
    """Keep the old bindings where names match; remap obvious equivalents; drop the rest."""
    if not brick.inputs:
        return  # inference models: port names are ours to choose
    keep: dict[str, Binding] = {}
    unused = [name for name in brick.inputs if name not in step.inputs]
    for param, binding in step.inputs.items():
        if param in brick.inputs:
            keep[param] = binding
            continue
        match = similarity_suggestions(param, unused)
        if match:
            keep[match[0]] = binding
            unused.remove(match[0])
    step.inputs = keep


def autofix_license(
    spec: WorkflowSpec,
    report: Report,
    registry: BrickRegistry,
) -> list[AutoFix]:
    """Swap a non-commercial brick for a commercial one when the workflow demands it."""
    fixes: list[AutoFix] = []
    for issue in report.issues:
        if issue.code != "LICENSE_NON_COMMERCIAL" or not issue.step_id:
            continue
        step = spec.step(issue.step_id)
        if step is None:
            continue
        current = registry.by_id(step.brick_id)
        alts = [registry.by_id(i) for i in (issue.data.get("alternatives") or [])]
        alts = [a for a in alts if a]
        if not alts and current is not None:
            # the validator may not have had a candidate list (e.g. offline validation)
            alts = registry.alternatives(current, commercial_only=True)
        target = next(
            (a for a in alts
             if a.commercial
             and a.status == "running"
             and a.output_kind == (current.output_kind if current else a.output_kind)),
            None,
        )
        if target is None:
            continue
        old = current.license if current else "unknown"
        step.brick_id = target.id
        step.source = target.source
        step.api_name = target.api_name
        step.kind = "inference" if target.kind == "inference_model" else "space"
        _rewire_after_swap(step, target)
        fixes.append(AutoFix(
            kind="license_swap",
            step_id=step.id,
            detail=f"'{old}' cannot be used commercially; switched to {target.source} "
                   f"({target.license})",
            before=old,
            after=target.license,
            issue_codes=[issue.code],
        ))
    return fixes


def run_autofixes(
    spec: WorkflowSpec,
    report: Report,
    registry: BrickRegistry,
) -> list[AutoFix]:
    """All deterministic fixes, in dependency order (brick swaps first: they rewrite params)."""
    fixes: list[AutoFix] = []
    fixes += autofix_broken_bricks(spec, report, registry)
    fixes += autofix_license(spec, report, registry)
    fixes += autofix_renamed_params(spec, report, registry)
    fixes += autofix_missing_params(spec, report, registry)
    return fixes