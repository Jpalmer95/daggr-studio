"""
Validation: turn a spec into a list of machine-readable findings.

Every finding has a stable ``code`` so the Medic (and the tests) can reason about classes
of failure rather than free text. Severity contract:

* ``blocking`` - the graph cannot run correctly; heal or refuse.
* ``warning``  - it will run but is fragile, slow, or off-policy.
* ``info``     - worth surfacing (cost, licensing nuance, token requirements).

Checks come in three layers:
1. structural      - offline, from ``WorkflowSpec.validate_shape``
2. registry        - offline, from the brick catalogue (licence, status, cost, fn params)
3. live            - network (cached) introspection of the real Space endpoints
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Literal

from daggrstudio.codegen.fn_library import fn_signatures
from daggrstudio.introspect import Param, SpaceInfo, introspect
from daggrstudio.registry.bricks import Brick, BrickRegistry, get_registry
from daggrstudio.spec import Binding, Step, WorkflowSpec

Severity = Literal["blocking", "warning", "info"]

MEDIA_KINDS = ("image", "audio", "video", "model3d", "file")
MAX_USEFUL_STEPS = 12


@dataclass
class Issue:
    code: str
    severity: Severity
    message: str
    step_id: str | None = None
    port: str | None = None
    fix_hint: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return self.severity == "blocking"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "step_id": self.step_id,
            "port": self.port,
            "fix_hint": self.fix_hint,
            "data": self.data,
        }

    def __str__(self) -> str:  # pragma: no cover - presentation only
        where = f" [{self.step_id}.{self.port}]" if self.step_id else ""
        return f"{self.severity.upper():8s} {self.code}{where}: {self.message}"


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    introspected: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.is_blocking]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def codes(self) -> list[str]:
        return sorted({i.code for i in self.blocking})

    def by_step(self, step_id: str) -> list[Issue]:
        return [i for i in self.issues if i.step_id == step_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "blocking": [i.to_dict() for i in self.blocking],
            "warnings": [i.to_dict() for i in self.warnings],
            "info": [i.to_dict() for i in self.issues if i.severity == "info"],
            "codes": self.codes(),
        }

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "All checks passed."
        bits = []
        if self.blocking:
            bits.append(f"{len(self.blocking)} blocking ({', '.join(self.codes())})")
        if self.warnings:
            bits.append(f"{len(self.warnings)} warning")
        info = [i for i in self.issues if i.severity == "info"]
        if info:
            bits.append(f"{len(info)} info")
        return " | ".join(bits)


# ─── shape problem -> coded issue ─────────────────────────────────────────────

_SHAPE_CODES = (
    ("no steps", "NO_STEPS", "blocking"),
    ("duplicate step id", "DUPLICATE_STEP_ID", "blocking"),
    ("missing an id", "STEP_ID_MISSING", "blocking"),
    ("unknown kind", "UNKNOWN_STEP_KIND", "blocking"),
    ("unknown fn", "UNKNOWN_FN", "blocking"),
    ("neither brick_id nor source", "STEP_SOURCE_MISSING", "blocking"),
    ("declares no output ports", "NO_OUTPUT_PORTS", "blocking"),
    ("undeclared input", "UNDECLARED_USER_INPUT", "blocking"),
    ("malformed edge", "MALFORMED_EDGE", "blocking"),
    ("references unknown step", "UNKNOWN_UPSTREAM_STEP", "blocking"),
    ("only exposes", "UNKNOWN_PORT_ON_UPSTREAM", "blocking"),
    ("unknown callable", "UNKNOWN_CALLABLE", "blocking"),
    ("cycle detected", "CYCLE", "blocking"),
    ("no inputs wired", "NO_INPUTS_WIRED", "blocking"),
    ("unknown component", "UNKNOWN_COMPONENT", "blocking"),
    ("unknown license_posture", "UNKNOWN_LICENSE_POSTURE", "warning"),
    ("input port missing a name", "INPUT_PORT_UNNAMED", "blocking"),
)


def _shape_issues(spec: WorkflowSpec) -> list[Issue]:
    issues = []
    for problem in spec.validate_shape():
        code, severity = "SHAPE", "blocking"
        for marker, candidate, sev in _SHAPE_CODES:
            if marker in problem:
                code, severity = candidate, sev
                break
        step_id = None
        for step in spec.steps:
            if f"'{step.id}'" in problem:
                step_id = step.id
                break
        issues.append(Issue(code=code, severity=severity, message=problem, step_id=step_id))
    return issues


# ── layer 2: registry ────────────────────────────────────────────────────────


def _registry_issues(spec: WorkflowSpec, registry: BrickRegistry) -> list[Issue]:
    issues: list[Issue] = []
    commercial_only = spec.license_posture == "commercial-only"

    for step in spec.steps:
        if step.kind == "fn":
            issues.extend(_fn_issues(step))
            continue
        if step.kind == "inference":
            # provider models are addressed by id, so a brick is nice-to-have not required
            brick = registry.by_id(step.brick_id) or registry.by_source(step.source or "")
            if brick is None:
                issues.append(Issue(
                    code="UNVERIFIED_MODEL", severity="warning",
                    message=f"model '{step.source}' is not in the verified brick registry",
                    step_id=step.id,
                    fix_hint="pick a registry model so licence and availability are known",
                ))
            continue

        brick = registry.by_id(step.brick_id) or (
            registry.by_source(step.source or "", step.api_name) if step.source else None
        )
        if brick is None:
            issues.append(Issue(
                code="UNKNOWN_BRICK", severity="blocking",
                message=(
                    f"brick '{step.brick_id or step.source}' is not in the registry "
                    f"(known ids: {len(registry)})"
                ),
                step_id=step.id,
                fix_hint="replace with a registry brick id that matches this modality",
                data={"known_ids": [b.id for b in registry.all()][:60]},
            ))
            continue

        if brick.status == "error":
            issues.append(Issue(
                code="BRICK_BROKEN", severity="blocking",
                message=f"brick '{brick.id}' ({brick.source}) failed verification: "
                        f"{brick.runtime_error}",
                step_id=step.id,
                fix_hint="replace with an alternative brick of the same modality",
                data={"modality": brick.modality,
                      "alternatives": [b.id for b in registry.alternatives(brick)][:8]},
            ))
        elif brick.status == "sleeping":
            issues.append(Issue(
                code="BRICK_SLEEPING", severity="info",
                message=f"brick '{brick.id}' is asleep; the first call may take 30-60s",
                step_id=step.id,
            ))

        if commercial_only and not brick.commercial:
            alts = [b.id for b in registry.alternatives(brick, commercial_only=True)][:8]
            issues.append(Issue(
                code="LICENSE_NON_COMMERCIAL", severity="blocking",
                message=(
                    f"brick '{brick.id}' is licensed '{brick.license}', which does not clearly "
                    f"permit commercial output, but this workflow is commercial-only"
                ),
                step_id=step.id,
                fix_hint="swap to a commercially licensed alternative, or set the workflow "
                         "licence posture to 'any' if NC output is acceptable",
                data={"license": brick.license, "alternatives": alts},
            ))

        if not brick.api_name and brick.kind == "space":
            issues.append(Issue(
                code="API_NAME_UNVERIFIED", severity="blocking",
                message=f"brick '{brick.id}' has no verified api_name (never introspected)",
                step_id=step.id,
                fix_hint="run the registry verifier, or pick a brick with api_name set",
            ))

        if step.api_name and brick.api_name and step.api_name != brick.api_name:
            issues.append(Issue(
                code="API_NAME_MISMATCH", severity="blocking",
                message=f"step calls '{step.api_name}' but the registry verified "
                        f"'{brick.api_name}' for {brick.source}",
                step_id=step.id,
                fix_hint=f"use api_name '{brick.api_name}'",
            ))

        if brick.notes and "token" in brick.notes.lower() and "ZeroGPU" in brick.notes:
            issues.append(Issue(
                code="NEEDS_HF_TOKEN", severity="info",
                message=f"brick '{brick.id}' is likely to need an HF token at run time",
                step_id=step.id,
            ))

    # cost tier honesty
    heavy = [s.id for s in spec.steps
             if (b := registry.by_id(s.brick_id)) and b.modality in ("video", "image-to-3d", "text-to-3d")]
    if heavy and spec.compute_tier == "cloud-free":
        issues.append(Issue(
            code="COST_TIER_OPTIMISTIC", severity="warning",
            message=f"workflow claims cloud-free but steps {heavy} are GPU-heavy or billable",
            fix_hint="set compute_tier to 'cloud-paid'",
        ))

    issues.extend(_graph_policy_issues(spec))
    return issues


def _fn_issues(step: Step) -> list[Issue]:
    issues: list[Issue] = []
    sigs = fn_signatures()
    params = sigs.get(step.fn or "")
    if params is None:
        return [Issue(code="UNKNOWN_FN", severity="blocking",
                      message=f"unknown helper function '{step.fn}'", step_id=step.id)]
    for port in step.inputs:
        if port not in params:
            issues.append(Issue(
                code="FN_PARAM_UNKNOWN", severity="blocking",
                message=f"helper '{step.fn}' has no parameter '{port}'",
                step_id=step.id, port=port,
                fix_hint=f"use one of {sorted(params)}", data={"params": sorted(params)},
            ))
    return issues


def _graph_policy_issues(spec: WorkflowSpec) -> list[Issue]:
    issues: list[Issue] = []
    referenced = set()
    for step in spec.steps:
        for binding in step.inputs.values():
            ref = binding.step_ref
            if ref:
                referenced.add(ref[0])

    for step in spec.steps:
        is_terminal = not any(
            b.step_ref and b.step_ref[0] == step.id
            for s in spec.steps for b in s.inputs.values()
        )
        if not is_terminal and step.id not in referenced and step.kind != "fn":
            issues.append(Issue(
                code="ORPHAN_STEP", severity="warning",
                message=f"step '{step.id}' feeds nothing downstream",
                step_id=step.id,
            ))

    used_inputs = set(spec.inputs_used())
    for port in spec.inputs:
        if port.port not in used_inputs:
            issues.append(Issue(
                code="UNUSED_INPUT", severity="warning",
                message=f"input '{port.port}' is declared but never used",
                port=port.port,
                fix_hint="wire it into a step or drop it",
            ))

    if len(spec.steps) > MAX_USEFUL_STEPS:
        issues.append(Issue(
            code="TOO_MANY_STEPS", severity="warning",
            message=f"{len(spec.steps)} steps is a lot for one canvas (> {MAX_USEFUL_STEPS})",
        ))

    bricks = [s.brick_id for s in spec.steps if s.brick_id]
    for dup in {b for b in bricks if bricks.count(b) > 1}:
        issues.append(Issue(
            code="DUPLICATE_BRICK", severity="info",
            message=f"brick '{dup}' is used more than once (fine, but check it is intended)",
        ))
    return issues


# ─── layer 3: live introspection ──────────────────────────────────────────────


def _live_issues(
    spec: WorkflowSpec,
    registry: BrickRegistry,
    token: str | None,
    report: Report,
) -> list[Issue]:
    issues: list[Issue] = []
    for step in spec.steps:
        if step.kind not in ("space", "inference"):
            continue
        brick = registry.by_id(step.brick_id) or registry.by_source(step.source or "")
        source, api_name = (
            (brick.source, brick.api_name) if brick else (step.source or "", step.api_name)
        )
        if not source:
            continue
        if step.kind == "inference":
            continue  # provider models are validated by the router, not view_api

        info = introspect(source, api_name, token=token)
        report.introspected[step.id] = {
            "source": source,
            "api_name": api_name,
            "ok": info.ok,
            "error": info.error,
            "params": info.param_names,
            "endpoints": sorted(info.endpoints),
        }
        if info.error_class == "EndpointMissing":
            # the Space answered, but our endpoint name is wrong - the classic "the sister
            # Space was updated and broke my workflow" case
            issues.append(Issue(
                code="API_NAME_INVALID", severity="blocking",
                message=f"{source}: {info.error}",
                step_id=step.id,
                fix_hint=f"available endpoints: {sorted(info.endpoints)}",
                data={"error_class": info.error_class, "endpoints": sorted(info.endpoints)},
            ))
            continue
        if not info.ok:
            issues.append(Issue(
                code="SPACE_UNREACHABLE", severity="blocking",
                message=f"{source}: {info.error}",
                step_id=step.id,
                fix_hint=(
                    f"available endpoints: {sorted(info.endpoints)}" if info.endpoints
                    else "the Space may be private, down, or require an HF token"
                ),
                data={"error_class": info.error_class, "endpoints": sorted(info.endpoints)},
            ))
            continue

        params = info.params
        if not params:
            continue
        known = {p.name for p in params}

        # (a) params the spec wires that the Space no longer accepts -> renamed upstream
        for port in step.inputs:
            if port in known:
                continue
            close = difflib.get_close_matches(port, sorted(known), n=3, cutoff=0.4)
            issues.append(Issue(
                code="PARAM_RENAMED",
                severity="blocking",
                message=f"{source}{api_name} does not accept '{port}' (live params: {sorted(known)})",
                step_id=step.id,
                port=port,
                fix_hint=(f"closest live match: '{close[0]}'" if close
                          else "this parameter may have been removed by the Space author"),
                data={"live_params": sorted(known), "suggestions": close},
            ))

        # (b) required params the spec never wires (would fail at call time)
        required = [p for p in params if not p.has_default and not p.is_file and not p.name.startswith("_")]
        wired = set(step.inputs)
        for param in required:
            if param.name in wired:
                continue
            # daggr itself refuses to build a node with a missing required parameter, so this
            # is blocking: the Medic must either wire it or pick a brick that does not need it.
            issues.append(Issue(
                code="PARAM_MISSING",
                severity="blocking",
                message=f"{source}{api_name} requires '{param.name}' ({param.type}) with no "
                        f"default, and nothing is wired to it",
                step_id=step.id,
                port=param.name,
                fix_hint=f"wire '{param.name}' (type {param.type}) with a literal, an upstream "
                         f"step, or a declared input - daggr will not build it otherwise",
                data={"live_params": sorted(known),
                      "modality": (brick.modality if brick else "unknown")},
            ))

        # (c) edge sanity: a media value into a numeric slot (or vice versa)
        for port, binding in step.inputs.items():
            ref = binding.step_ref
            if not ref:
                continue
            upstream = spec.step(ref[0])
            if upstream is None:
                continue
            up_kind = _output_kind_of(upstream, registry)
            target = next((p for p in params if p.name == port), None)
            if target is None:
                continue
            if up_kind in MEDIA_KINDS and target.is_scalar:
                issues.append(Issue(
                    code="EDGE_TYPE_MISMATCH", severity="blocking",
                    message=f"{upstream.id} outputs {up_kind} but {source} param '{port}' "
                            f"is scalar ({target.type})",
                    step_id=step.id, port=port,
                    fix_hint="insert/adjust a step, or wire a text/number output instead",
                ))
            elif up_kind in ("text", "json") and target.is_file:
                issues.append(Issue(
                    code="EDGE_TYPE_MISMATCH", severity="blocking",
                    message=f"{upstream.id} outputs {up_kind} but {source} param '{port}' "
                            f"expects a file ({target.type})",
                    step_id=step.id, port=port,
                    fix_hint="add a brick that produces a file, or fix the wiring",
                ))
    return issues


def _output_kind_of(step: Step, registry: BrickRegistry) -> str:
    brick = registry.by_id(step.brick_id)
    if brick:
        return brick.output_kind
    for meta in step.outputs.values():
        comp = (meta.get("component") or "").lower()
        if comp in MEDIA_KINDS:
            return comp
        if comp == "textbox":
            return "text"
        if comp == "json":
            return "json"
    return "unknown"


# ─── public entry point ───────────────────────────────────────────────────────


def validate(
    spec: WorkflowSpec,
    registry: BrickRegistry | None = None,
    live: bool = True,
    token: str | None = None,
) -> Report:
    """
    Validate a spec. ``live=False`` skips network introspection (fast, offline, used in
    tests and in the healing loop once endpoints are already known).
    """
    registry = registry or get_registry()
    report = Report()
    report.issues.extend(_shape_issues(spec))
    report.issues.extend(_registry_issues(spec, registry))
    if live:
        report.issues.extend(_live_issues(spec, registry, token, report))
    return report


def similarity_suggestions(port: str, known: list[str], n: int = 3) -> list[str]:
    return difflib.get_close_matches(port, known, n=n, cutoff=0.4)