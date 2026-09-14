"""
The patch language.

The Medic never rewrites a spec. It emits a list of *operations* from a small, closed
vocabulary, which are applied deterministically. Anything unrecognised is rejected and
reported back to the model on the next round - so a sloppy repair degrades into "no
change", never into a broken spec.

Keeping the vocabulary small is what makes healing safe: every op is unit-tested, and the
worst a model can do is pick a different verified brick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn

from daggrstudio.registry.bricks import BrickRegistry
from daggrstudio.spec import Binding, COMPONENTS, InputPort, Step, WorkflowSpec

OPS = (
    "set_input",       # wire/replace one input param binding of a step
    "drop_input",      # remove a param the Space no longer accepts
    "set_api_name",    # fix the endpoint a step calls
    "replace_brick",   # swap the model/Space for another verified brick
    "set_outputs",     # fix output port declarations
    "set_postprocess", # fix how a multi-value return is unpacked
    "add_step",        # insert a missing pre/post-processing step
    "remove_step",     # drop a step
    "set_meta",        # name / industry / tags / license_posture / compute_tier / notes
    "declare_inputs",  # add or fix the user-facing inputs
    "set_concurrent",  # toggle FnNode concurrency
)

#: meta fields the model may change, and the value type each expects
META_FIELDS = {
    "name": str,
    "intent": str,
    "industry": str,
    "tags": list,
    "license_posture": str,
    "compute_tier": str,
    "notes": str,
}


@dataclass
class PatchResult:
    applied: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    def summary(self) -> str:
        if not self.applied and not self.skipped:
            return "no operations"
        bits = []
        if self.applied:
            bits.append(
                "applied: " + ", ".join(f"{op['op']}({op.get('step', '')})" for op in self.applied)
            )
        if self.skipped:
            bits.append(
                "skipped: " + ", ".join(f"{op.get('op')}({op.get('reason', '')})" for op in self.skipped)
            )
        return " | ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {"applied": self.applied, "skipped": self.skipped, "summary": self.summary()}


def _skip(op: dict[str, Any], reason: str) -> NoReturn:
    op = dict(op)
    op["reason"] = reason
    raise _SkipOp(op)


class _SkipOp(Exception):
    """Internal: one bad op must not abort the whole patch batch."""

    def __init__(self, op: dict[str, Any]):
        super().__init__(op.get("reason", ""))
        self.op = op


def normalise_ops(payload: Any) -> list[dict[str, Any]]:
    """Accept {"patches": [...]}, {"ops": [...]}, a bare list, or a single op dict."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("patches", "ops", "operations", "edits"):
            if isinstance(payload.get(key), list):
                return [p for p in payload[key] if isinstance(p, dict)]
        if "op" in payload:
            return [payload]
        return []
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    return []


def apply_ops(
    spec: WorkflowSpec,
    ops: list[dict[str, Any]],
    registry: BrickRegistry,
) -> PatchResult:
    """
    Apply operations to ``spec`` in place. Returns what happened; never raises.

    Invalid ops are collected in ``skipped`` with a reason so the next Medic round (or the
    user) can see exactly why a suggested repair was not taken.
    """
    result = PatchResult()
    for op in ops:
        try:
            _apply_one(spec, op, registry)
            result.applied.append({k: v for k, v in op.items() if k != "reason"})
        except _SkipOp as exc:
            result.skipped.append(exc.op)
        except Exception as exc:  # malformed op -> report, keep going
            bad = dict(op)
            bad["reason"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            result.skipped.append(bad)
    if result.changed:
        spec.touch()
    return result


def _apply_one(spec: WorkflowSpec, op: dict[str, Any], registry: BrickRegistry) -> None:
    name = op.get("op")
    if name not in OPS:
        _skip(op, f"unknown op '{name}' (allowed: {list(OPS)})")

    step_id = op.get("step") or op.get("step_id") or op.get("node")
    step = spec.step(str(step_id)) if step_id else None
    if name in ("set_input", "drop_input", "set_api_name", "replace_brick", "set_outputs",
                "set_postprocess", "remove_step", "set_concurrent") and step is None:
        _skip(op, f"no step with id '{step_id}'")

    if name == "set_input":
        param = op.get("param") or op.get("name") or op.get("port")
        if not param:
            _skip(op, "set_input needs 'param'")
        binding_payload = op.get("binding", op.get("value"))
        if isinstance(binding_payload, dict) and ("from" in binding_payload or "callable" in binding_payload
                                                  or "value" in binding_payload):
            binding = Binding.from_dict(binding_payload)
        elif "from" in op:
            binding = Binding(source=str(op["from"]))
        else:
            binding = Binding(value=binding_payload)
        # reject edges pointing at nothing
        ref = binding.step_ref
        if binding.kind == "step" and ref:
            upstream = spec.step(ref[0])
            if upstream is None:
                _skip(op, f"edge references unknown step '{ref[0]}'")
            if ref[1] not in upstream.output_ports:
                _skip(op, f"step '{ref[0]}' has no output port '{ref[1]}'")
        if binding.kind == "user" and binding.user_port not in {i.port for i in spec.inputs}:
            _skip(op, f"input '{binding.user_port}' is not declared")
        step.inputs[str(param)] = binding  # type: ignore[union-attr]
        return

    if name == "drop_input":
        param = op.get("param") or op.get("name") or op.get("port")
        if param in step.inputs:  # type: ignore[union-attr]
            del step.inputs[str(param)]  # type: ignore[union-attr]
        else:
            _skip(op, f"step has no input '{param}'")
        return

    if name == "set_api_name":
        api = op.get("api_name") or op.get("endpoint")
        if not api or not str(api).startswith("/"):
            _skip(op, "api_name must look like '/endpoint'")
        step.api_name = str(api)  # type: ignore[union-attr]
        return

    if name == "replace_brick":
        brick_id = op.get("brick_id") or op.get("brick")
        brick = registry.by_id(brick_id)
        if brick is None:
            _skip(op, f"'{brick_id}' is not a verified brick")
        step.brick_id = brick.id  # type: ignore[union-attr]
        step.source = brick.source  # type: ignore[union-attr]
        step.api_name = brick.api_name  # type: ignore[union-attr]
        step.kind = "inference" if brick.kind == "inference_model" else "space"  # type: ignore[union-attr]
        # keep only params the new brick accepts (the Medic re-wires what it wants after)
        keep = set(brick.inputs) if brick.inputs else set(step.inputs)  # type: ignore[union-attr]
        dropped = [p for p in step.inputs if p not in keep]  # type: ignore[union-attr]
        for param in dropped:
            if step.inputs[param].kind == "step":  # type: ignore[union-attr]
                continue  # never silently break an edge
            step.inputs.pop(param)  # type: ignore[union-attr]
        return

    if name == "set_outputs":
        outputs = op.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            _skip(op, "set_outputs needs a non-empty 'outputs' mapping")
        clean: dict[str, dict[str, Any]] = {}
        for port, meta in outputs.items():
            if isinstance(meta, str):
                meta = {"component": meta}
            if not isinstance(meta, dict):
                _skip(op, f"output '{port}' must be a component or {{component: ...}}")
            component = str(meta.get("component", "textbox")).lower()
            if component not in COMPONENTS:
                _skip(op, f"unknown component '{component}'")
            clean[str(port)] = {"component": component, "label": str(meta.get("label", ""))}
        step.outputs = clean  # type: ignore[union-attr]
        return

    if name == "set_postprocess":
        hint = op.get("hint") or op.get("postprocess")
        if hint is None:
            step.postprocess = None  # type: ignore[union-attr]
            return
        hint = str(hint).strip()
        ok = (hint in ("first", "keep_first", "unwrap_path")
              or hint.startswith("tuple_index:") or hint.startswith("dict_path:"))
        if not ok:
            _skip(op, f"unrecognised postprocess hint '{hint}'")
        step.postprocess = hint  # type: ignore[union-attr]
        return

    if name == "set_concurrent":
        step.concurrent = bool(op.get("concurrent", op.get("value", True)))  # type: ignore[union-attr]
        return

    if name == "remove_step":
        refd = [
            f"{s.id}.{port}"
            for s in spec.steps
            for port, b in s.inputs.items()
            if (b.step_ref or ("", ""))[0] == step.id  # type: ignore[union-attr]
        ]
        if refd:
            _skip(op, f"'{step.id}' is still referenced by {refd[:3]}")
        spec.steps = [s for s in spec.steps if s.id != step.id]
        return

    if name == "add_step":
        payload = op.get("step") if isinstance(op.get("step"), dict) else op.get("new_step")
        if not isinstance(payload, dict):
            _skip(op, "add_step needs a 'step' object")
        new_step = Step.from_dict(payload)
        if not new_step.id:
            _skip(op, "new step needs an id")
        if spec.step(new_step.id):
            _skip(op, f"step id '{new_step.id}' already exists")
        if new_step.kind in ("space", "inference") and not registry.by_id(new_step.brick_id):
            _skip(op, f"new step brick '{new_step.brick_id}' is not verified")
        spec.steps.append(new_step)
        return

    if name == "set_meta":
        changed = False
        for key, expected in META_FIELDS.items():
            if key not in op:
                continue
            value = op[key]
            if not isinstance(value, expected):
                _skip(op, f"meta '{key}' must be {expected.__name__}")
            if key == "license_posture" and value not in ("commercial-only", "any"):
                _skip(op, "license_posture must be 'commercial-only' or 'any'")
            setattr(spec, key, value)
            changed = True
        if not changed:
            _skip(op, f"set_meta accepts one of {sorted(META_FIELDS)}")
        return

    if name == "declare_inputs":
        payload = op.get("inputs")
        if not isinstance(payload, list) or not payload:
            _skip(op, "declare_inputs needs a non-empty 'inputs' list")
        ports = [InputPort.from_dict(p) for p in payload]
        if any(not p.port for p in ports):
            _skip(op, "every declared input needs a 'port'")
        bad = [p.port for p in ports if p.component not in COMPONENTS]
        if bad:
            _skip(op, f"unknown component for inputs {bad}")
        spec.inputs = ports
        return

    _skip(op, f"op '{name}' not implemented")