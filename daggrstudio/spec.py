"""
WorkflowSpec — the intermediate representation between "what the user wants" and
"real daggr code".

Design rule: the LLM only ever authors a *spec* (JSON), never Python. Everything that
turns a spec into a running graph is deterministic and unit-tested (`codegen.py`).
That makes repairs safe: the Medic patches the spec, and the code is regenerated.

A spec is deliberately boring JSON so it can be saved, shared, diffed, and versioned in
a Hugging Face Dataset.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1

MODALITIES = [
    "text",
    "image-gen",
    "image-edit",
    "image-to-3d",
    "text-to-3d",
    "video",
    "audio-tts",
    "music",
    "vision",
    "upscale",
    "utility",
]

INDUSTRIES = [
    "game-dev",
    "music-production",
    "art",
    "3d",
    "film",
    "marketing",
    "general",
]

STEP_KINDS = ("space", "inference", "fn")

#: Gradio components an input or output port may use.
COMPONENTS = (
    "textbox",
    "image",
    "audio",
    "video",
    "model3d",
    "number",
    "slider",
    "dropdown",
    "checkbox",
    "json",
    "gallery",
    "file",
)

#: Named zero-argument callables the planner may use for hidden input values.
CALLABLES = ("random_int", "random_float", "uuid4", "now_iso", "random_choice_style")

#: Named local functions an `fn` step may reference (see codegen.fn_library).
FN_NAMES = (
    "join_texts",
    "text_to_lines",
    "summarize_stats",
    "organize_outputs",
    "pick_first",
    "json_report",
)

LICENSE_POSTURES = ("commercial-only", "any")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:60] or "workflow"


# ─── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class InputPort:
    """A user-facing input rendered as a Gradio component in the canvas."""

    port: str
    component: str = "textbox"
    label: str = ""
    default: Any = ""
    lines: int = 3
    choices: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    info: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"port": self.port, "component": self.component}
        if self.label:
            d["label"] = self.label
        if self.default not in ("", None):
            d["default"] = self.default
        if self.component == "textbox" and self.lines:
            d["lines"] = self.lines
        if self.choices:
            d["choices"] = list(self.choices)
        for key in ("minimum", "maximum", "step"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        if self.info:
            d["info"] = self.info
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InputPort":
        if isinstance(d, str):  # tolerate "prompt" shorthand
            return cls(port=d, label=d.replace("_", " ").title())
        return cls(
            port=str(d.get("port") or d.get("name") or ""),
            component=str(d.get("component", "textbox")),
            label=str(d.get("label", "")),
            default=d.get("default", ""),
            lines=int(d.get("lines", 3) or 3),
            choices=list(d.get("choices") or []),
            minimum=d.get("minimum"),
            maximum=d.get("maximum"),
            step=d.get("step"),
            info=str(d.get("info", "")),
        )


@dataclass
class Binding:
    """
    How one input parameter of a step gets its value.

    Exactly one of the four shapes:
      * ``{"from": "user:<port>"}``            — a UI input
      * ``{"from": "step:<step_id>.<port>"}``  — an edge from an upstream step
      * ``{"value": <json>}``                  — a fixed constant (hidden in the canvas)
      * ``{"callable": "<name>"}``             — evaluated on every run (hidden)
    """

    source: str | None = None
    value: Any = None
    callable: str | None = None

    @property
    def kind(self) -> str:
        if self.source:
            return "user" if self.source.startswith("user:") else "step"
        if self.callable:
            return "callable"
        return "value"

    @property
    def user_port(self) -> str | None:
        return self.source[5:] if self.source and self.source.startswith("user:") else None

    @property
    def step_ref(self) -> tuple[str, str] | None:
        """("step_id", "port") when this binding is an edge, else None."""
        if not self.source or not self.source.startswith("step:"):
            return None
        body = self.source[5:]
        if "." not in body:
            return None
        step_id, _, port = body.partition(".")
        return step_id, port

    def to_dict(self) -> dict[str, Any]:
        if self.source:
            return {"from": self.source}
        if self.callable:
            return {"callable": self.callable}
        return {"value": self.value}

    @classmethod
    def from_dict(cls, d: Any) -> "Binding":
        if isinstance(d, Binding):
            return d
        if isinstance(d, dict):
            if "from" in d:
                return cls(source=str(d["from"]))
            if "callable" in d:
                return cls(callable=str(d["callable"]))
            if "value" in d:
                return cls(value=d["value"])
        return cls(value=d)


@dataclass
class Step:
    id: str
    kind: str = "space"
    brick_id: str | None = None
    source: str | None = None  # denormalised owner/name; registry is authoritative
    api_name: str | None = None
    title: str = ""
    why: str = ""
    inputs: dict[str, Binding] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    postprocess: str | None = None
    fn: str | None = None  # for kind="fn"
    params: dict[str, Any] = field(default_factory=dict)
    concurrent: bool = True
    concurrency_group: str | None = None

    @property
    def output_ports(self) -> list[str]:
        return list(self.outputs.keys()) or (["output"] if self.kind == "fn" else [])

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "kind": self.kind}
        for key in ("title", "why", "brick_id", "source", "api_name", "postprocess", "fn"):
            val = getattr(self, key)
            if val:
                d[key] = val
        d["inputs"] = {k: v.to_dict() for k, v in self.inputs.items()}
        d["outputs"] = copy.deepcopy(self.outputs)
        if self.concurrency_group:
            d["concurrency_group"] = self.concurrency_group
        if self.kind == "fn":
            d["concurrent"] = self.concurrent
            if self.params:
                d["params"] = dict(self.params)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        return cls(
            id=str(d.get("id") or ""),
            kind=str(d.get("kind", "space")),
            brick_id=d.get("brick_id"),
            source=d.get("source"),
            api_name=d.get("api_name"),
            title=str(d.get("title", "")),
            why=str(d.get("why", "")),
            inputs={k: Binding.from_dict(v) for k, v in (d.get("inputs") or {}).items()},
            outputs={
                k: (v if isinstance(v, dict) else {"component": str(v)})
                for k, v in (d.get("outputs") or {}).items()
            },
            postprocess=d.get("postprocess"),
            fn=d.get("fn"),
            params=dict(d.get("params") or {}),
            concurrent=bool(d.get("concurrent", True)),
            concurrency_group=d.get("concurrency_group"),
        )

    def clone(self) -> "Step":
        return Step.from_dict(copy.deepcopy(self.to_dict()))


@dataclass
class WorkflowSpec:
    name: str = "Untitled workflow"
    intent: str = ""
    steps: list[Step] = field(default_factory=list)
    inputs: list[InputPort] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    id: str = ""
    slug: str = ""
    industry: str = "general"
    tags: list[str] = field(default_factory=list)
    license_posture: str = "commercial-only"
    compute_tier: str = "cloud-free"
    notes: str = ""
    author: str = ""
    created_at: str = ""
    updated_at: str = ""
    planner_model: str = ""
    heal_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = slugify(self.name)
        if not self.slug:
            self.slug = self.id
        if not self.created_at:
            self.created_at = _now()
        self.updated_at = self.updated_at or self.created_at

    # ── convenience ─────────────────────────────────────────────────────────────

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    @property
    def brick_ids(self) -> list[str]:
        return [s.brick_id for s in self.steps if s.brick_id]

    @property
    def edges(self) -> list[tuple[str, str, str, str]]:
        """(from_step, from_port, to_step, to_port) for every step→step binding."""
        out = []
        for step in self.steps:
            for port, binding in step.inputs.items():
                ref = binding.step_ref
                if ref:
                    out.append((ref[0], ref[1], step.id, port))
        return out

    def inputs_used(self) -> list[str]:
        used = []
        for step in self.steps:
            for binding in step.inputs.values():
                up = binding.user_port
                if up and up not in used:
                    used.append(up)
        return used

    # ── serialization ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "slug": self.slug,
            "name": self.name,
            "intent": self.intent,
            "industry": self.industry,
            "tags": list(self.tags),
            "license_posture": self.license_posture,
            "compute_tier": self.compute_tier,
            "notes": self.notes,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "planner_model": self.planner_model,
            "inputs": [i.to_dict() for i in self.inputs],
            "steps": [s.to_dict() for s in self.steps],
            "heal_log": copy.deepcopy(self.heal_log),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowSpec":
        inputs = d.get("inputs") or []
        if isinstance(inputs, dict):  # tolerate {"prompt": {...}}
            inputs = [dict(v, port=k) if isinstance(v, dict) else k for k, v in inputs.items()]
        spec = cls(
            name=str(d.get("name") or "Untitled workflow"),
            intent=str(d.get("intent", "")),
            steps=[Step.from_dict(s) for s in (d.get("steps") or [])],
            inputs=[InputPort.from_dict(i) for i in inputs],
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            id=str(d.get("id") or ""),
            slug=str(d.get("slug") or ""),
            industry=str(d.get("industry", "general")),
            tags=list(d.get("tags") or []),
            license_posture=str(d.get("license_posture", "commercial-only")),
            compute_tier=str(d.get("compute_tier", "cloud-free")),
            notes=str(d.get("notes", "")),
            author=str(d.get("author", "")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            planner_model=str(d.get("planner_model", "")),
            heal_log=list(d.get("heal_log") or []),
        )
        return spec

    def clone(self) -> "WorkflowSpec":
        return WorkflowSpec.from_dict(copy.deepcopy(self.to_dict()))

    def touch(self) -> None:
        self.updated_at = _now()

    # ── shape validation (structural only; live checks live in validator.py) ────

    def validate_shape(self) -> list[str]:
        """Return human-readable structural problems. Empty list == structurally sound."""
        problems: list[str] = []
        if not self.steps:
            problems.append("workflow has no steps")
        ids = [s.id for s in self.steps]
        for dup in {i for i in ids if ids.count(i) > 1}:
            problems.append(f"duplicate step id: {dup}")
        user_ports = {i.port for i in self.inputs}
        for inp in self.inputs:
            if not inp.port:
                problems.append("input port missing a name")
            elif inp.component not in COMPONENTS:
                problems.append(f"input '{inp.port}' uses unknown component '{inp.component}'")
        for step in self.steps:
            if not step.id:
                problems.append("step missing an id")
            if step.kind not in STEP_KINDS:
                problems.append(f"step '{step.id}' has unknown kind '{step.kind}'")
            if step.kind == "fn":
                if step.fn not in FN_NAMES:
                    problems.append(f"step '{step.id}' references unknown fn '{step.fn}'")
            else:
                if not step.brick_id and not step.source:
                    problems.append(f"step '{step.id}' has neither brick_id nor source")
            if not step.outputs and step.kind != "fn":
                problems.append(f"step '{step.id}' declares no output ports")
            for port, binding in step.inputs.items():
                if binding.kind == "user" and binding.user_port not in user_ports:
                    problems.append(
                        f"step '{step.id}' input '{port}' references undeclared input "
                        f"'{binding.user_port}'"
                    )
                ref = binding.step_ref
                if binding.source and binding.kind == "step":
                    if ref is None:
                        problems.append(
                            f"step '{step.id}' input '{port}' has malformed edge "
                            f"'{binding.source}' (want step:<id>.<port>)"
                        )
                    else:
                        upstream, upstream_port = ref
                        target = self.step(upstream)
                        if target is None:
                            problems.append(
                                f"step '{step.id}' input '{port}' references unknown step "
                                f"'{upstream}'"
                            )
                        elif upstream_port not in target.output_ports:
                            problems.append(
                                f"step '{step.id}' input '{port}' reads '{upstream_port}' but "
                                f"step '{upstream}' only exposes {target.output_ports}"
                            )
                if binding.callable and binding.callable not in CALLABLES:
                    problems.append(
                        f"step '{step.id}' input '{port}' uses unknown callable "
                        f"'{binding.callable}'"
                    )
        problems.extend(self._cycle_problems())
        for step in self.steps:
            if step.kind in ("space", "inference") and not step.inputs:
                problems.append(f"step '{step.id}' has no inputs wired")
        if self.license_posture not in LICENSE_POSTURES:
            problems.append(f"unknown license_posture '{self.license_posture}'")
        return problems

    def _cycle_problems(self) -> list[str]:
        from collections import defaultdict

        graph: dict[str, list[str]] = defaultdict(list)
        for src, _sp, dst, _dp in self.edges:
            graph[src].append(dst)
        seen: set[str] = set()
        stack: set[str] = set()
        problems: list[str] = []

        def walk(node: str) -> None:
            seen.add(node)
            stack.add(node)
            for nxt in graph.get(node, []):
                if nxt in stack:
                    problems.append(f"cycle detected through step '{nxt}'")
                elif nxt not in seen:
                    walk(nxt)
            stack.discard(node)

        for sid in [s.id for s in self.steps]:
            if sid not in seen:
                walk(sid)
        return problems

    def is_healthy(self) -> bool:
        return not self.validate_shape()


def spec_from_json(text: str) -> WorkflowSpec | None:
    """Parse a WorkflowSpec out of raw model output, or None if there is no JSON object."""
    from daggrstudio.jsonutil import parse_json_object

    data = parse_json_object(text)
    if not isinstance(data, dict):
        return None
    if "workflow" in data and isinstance(data["workflow"], dict):
        data = data["workflow"]
    if "steps" not in data:
        return None
    return WorkflowSpec.from_dict(data)

