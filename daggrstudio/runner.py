"""
Execution: run a generated graph in-process with daggr's own executor, repairing as it goes.

Two things this buys over a plain ``graph.launch()``:

1. **Why-style reporting.** We drive the graph node by node, so the UI can show which brick
   ran, what it produced and how long it took - and the user can see a failure at the exact
   step that caused it instead of one opaque traceback.
2. **Runtime repair.** A sister Space being asleep, rate-limited, renamed or deleted is not
   a dead end: each failure is classified, then repaired by the cheapest means that can
   work (retry -> token -> live re-introspection remap -> fail over to another brick), and
   the repair is recorded so the saved workflow reflects reality.

We use daggr's ``SequentialExecutor`` rather than reimplementing node calling, so execution
semantics (postprocess, file handling, port mapping) are exactly daggr's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from daggrstudio.codegen import build_graph
from daggrstudio.medic.autofix import _rewire_after_swap  # noqa: F401  (kept for parity)
from daggrstudio.registry.bricks import BrickRegistry, get_registry
from daggrstudio.spec import WorkflowSpec
from daggrstudio.validator import Issue, Report

#: Error classifications we know how to act on.
SLEEPING_MARKERS = ("is in the invalid state", "starting", "sleeping", "503", "loading",
                    "runtime_error", "no GPU", "restarting")
QUOTA_MARKERS = ("quota", "zerogpu", "gpu task", "exceeded", "rate limit", "429", "402")
SIGNATURE_MARKERS = ("unexpected keyword", "positional argument", "got an unexpected",
                     "missing 1 required", "typerror", "cannot unpack", "valueerror: parameter")
GONE_MARKERS = ("404", "repositorynotfound", "not found", "removed", "does not exist")


@dataclass
class StepRun:
    node: str
    step_id: str
    title: str = ""
    brick_id: str = ""
    source: str = ""
    status: str = "pending"   # pending | ok | failed | repaired | skipped
    seconds: float = 0.0
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str = ""
    error_class: str = ""
    repairs: list[str] = field(default_factory=list)
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "repaired")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "step_id": self.step_id,
            "title": self.title,
            "brick_id": self.brick_id,
            "source": self.source,
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "artifacts": self.artifacts,
            "error": self.error[:400],
            "error_class": self.error_class,
            "repairs": self.repairs,
            "attempts": self.attempts,
        }


@dataclass
class RunResult:
    ok: bool = False
    steps: list[StepRun] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    seconds: float = 0.0
    repaired_spec: WorkflowSpec | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "seconds": round(self.seconds, 2),
            "artifacts": self.artifacts,
            "note": self.note,
            "steps": [s.to_dict() for s in self.steps],
        }

    def summary(self) -> str:
        done = sum(1 for s in self.steps if s.ok)
        repaired = sum(1 for s in self.steps if s.repairs)
        bits = [f"{done}/{len(self.steps)} steps produced output",
                f"{len(self.artifacts)} artifact(s)",
                f"{self.seconds:.1f}s"]
        if repaired:
            bits.append(f"{repaired} step(s) self-repaired")
        return " | ".join(bits)


def classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(m in text for m in SIGNATURE_MARKERS):
        return "signature"
    if any(m in text for m in GONE_MARKERS):
        return "gone"
    if any(m in text for m in QUOTA_MARKERS):
        return "quota"
    if any(m in text for m in SLEEPING_MARKERS):
        return "sleeping"
    return "unknown"


def collect_artifacts(value: Any, into: list[str]) -> None:
    """Pick file paths out of whatever shape a node returned."""
    import os

    if isinstance(value, str):
        if value.startswith("/") or value.startswith("./") and ".png" in value.lower():
            if os.path.exists(value):
                into.append(value)
        elif os.path.exists(value):
            into.append(value)
    elif isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and os.path.exists(path):
            into.append(path)
        else:
            for item in value.values():
                collect_artifacts(item, into)
    elif isinstance(value, (list, tuple)):
        for item in value:
            collect_artifacts(item, into)


def _entry_inputs(spec: WorkflowSpec, values: dict[str, Any], inputs_node_name: str) -> dict[str, dict[str, Any]]:
    cleaned = {port: val for port, val in (values or {}).items() if val is not None}
    return {inputs_node_name: cleaned}


def run_spec(
    spec: WorkflowSpec,
    values: dict[str, Any] | None = None,
    token: str | None = None,
    registry: BrickRegistry | None = None,
    repair: bool = True,
    progress: Callable[[StepRun], None] | None = None,
    max_repairs: int = 3,
) -> RunResult:
    """
    Execute a workflow spec and return per-step results.

    ``repair=True`` enables runtime self-healing (retry, token, remap, brick failover).
    ``progress`` is called after each step so a UI can stream state.
    """
    registry = registry or get_registry()
    result = RunResult()
    started = time.time()

    try:
        graph = build_graph(spec, registry)
    except Exception as exc:
        result.note = f"could not build the graph: {type(exc).__name__}: {exc}"
        return result

    from daggr.executor import SequentialExecutor

    executor = SequentialExecutor(graph, hf_token=token)
    input_node_names = {
        name for name, node in graph.nodes.items() if type(node).__name__ == "InputNode"
    }
    order = list(graph.get_execution_order())
    by_name = {n.id: n for n in spec.steps}
    runs: dict[str, StepRun] = {}

    def make_run(node_name: str) -> StepRun:
        import re

        step_id = re.sub(r"_\d+$", "", node_name)
        step = by_name.get(step_id) or by_name.get(node_name)
        brick = registry.by_id(step.brick_id) if step else None
        return StepRun(
            node=node_name,
            step_id=step.id if step else node_name,
            title=(step.title or step.id) if step else node_name,
            brick_id=(step.brick_id or "") if step else "",
            source=(brick.source if brick else (step.source or "" if step else "")),
        )

    repairs_used = 0
    completed_results: dict[str, Any] = {}
    entry = _entry_inputs(spec, values or {}, "Inputs")
    for node_name in order:
        is_input_node = node_name in input_node_names
        if not is_input_node:
            run = make_run(node_name)
            runs[node_name] = run
            result.steps.append(run)
        else:
            run = StepRun(node=node_name, step_id=node_name, title="Inputs", status="pending")

        # daggr takes entry values on the InputNode only; everything downstream reads its
        # upstream values out of the session, so we must pass them to exactly that node.
        node_inputs = entry.get(node_name) if is_input_node else None
        attempt = 0
        while True:
            attempt += 1
            run.attempts = attempt
            t0 = time.time()
            try:
                executor.execute_node(node_name, node_inputs)
                run.seconds += time.time() - t0
                run.status = "repaired" if run.repairs else "ok"
                outputs = (executor.results or {}).get(node_name, {})
                run.outputs = outputs if isinstance(outputs, dict) else {"output": outputs}
                found: list[str] = []
                collect_artifacts(run.outputs, found)
                run.artifacts = sorted(set(found))
                result.artifacts.extend(run.artifacts)
                break
            except Exception as exc:
                run.seconds += time.time() - t0
                run.error = f"{type(exc).__name__}: {exc}"
                run.error_class = classify_error(exc)
                if not repair or repairs_used >= max_repairs:
                    run.status = "failed"
                    break

                healed = _attempt_repair(
                    spec=spec, run=run, registry=registry, token=token,
                    executor=executor, node_name=node_name, graph=graph,
                )
                if healed is None:
                    run.status = "failed"
                    break
                repairs_used += 1
                rebuilt, note = healed
                run.repairs.append(note)
                if rebuilt is not None:
                    graph = rebuilt
                    executor = SequentialExecutor(graph, hf_token=token)
                    # keep everything already computed so upstream work is not repeated
                    executor.results = dict(completed_results)
                else:
                    time.sleep(min(30, 3 * attempt))
                completed_results = dict(executor.results or {})

        if progress:
            progress(run)
        completed_results = dict(executor.results or {})

    result.ok = any(s.ok for s in result.steps) and not any(
        s.status == "failed" and not s.ok for s in result.steps
    )
    result.seconds = time.time() - started
    if not result.ok:
        failed = [s.step_id for s in result.steps if s.status == "failed"]
        result.note = f"failed at: {', '.join(failed)}" if failed else "no steps ran"
    return result


def _attempt_repair(
    spec: WorkflowSpec,
    run: StepRun,
    registry: BrickRegistry,
    token: str | None,
    executor: Any,
    node_name: str,
    graph: Any,
) -> tuple[Any, str] | None:
    """
    Try the cheapest applicable repair for one failed step.

    Returns ``(new_graph_or_None, human_readable_note)`` when a repair was attempted, or
    ``None`` when there is nothing sensible left to try.
    """
    step = spec.step(run.step_id) or spec.step(node_name)
    brick = registry.by_id(step.brick_id) if step else None
    klass = run.error_class

    if klass == "sleeping":
        return None, "retrying a sleeping Space with backoff"

    if klass == "quota":
        if token:
            executor.set_hf_token(token)
            return None, "retrying with your Hugging Face token (GPU quota)"
        return None, "free-tier GPU quota hit; add an HF token in Settings to continue"

    if step is None or brick is None:
        return None, "no repair available: the failing step is not a registry brick"

    if klass == "signature":
        # the Space is alive but changed its interface: re-read it and remap
        from daggrstudio.introspect import introspect

        info = introspect(brick.source, brick.api_name, force=True, token=token)
        if info.ok and info.params:
            live = [p.name for p in info.params]
            from daggrstudio.validator import similarity_suggestions

            moved = False
            for param in list(step.inputs):
                if param in live:
                    continue
                match = similarity_suggestions(param, live)
                if match and match[0] not in step.inputs:
                    step.inputs[match[0]] = step.inputs.pop(param)
                    moved = True
            if moved:
                return build_graph(spec, registry), (
                    f"{brick.source} changed its API; re-mapped parameters to {live[:6]}"
                )
        return None, "the Space changed its API in a way we cannot map automatically"

    if klass == "gone":
        candidates = [b for b in registry.alternatives(brick) if b.status == "running"
                      and b.output_kind == brick.output_kind and b.api_name]
        if candidates:
            target = candidates[0]
            step.brick_id = target.id
            step.source = target.source
            step.api_name = target.api_name
            step.kind = "inference" if target.kind == "inference_model" else "space"
            from daggrstudio.medic.autofix import _rewire_after_swap

            _rewire_after_swap(step, target)
            return build_graph(spec, registry), (
                f"{brick.source} is gone; switched to {target.source} ({target.id})"
            )
    return None, f"unrecognised failure ({klass}); left for the Medic"


def run_and_report(
    spec: WorkflowSpec,
    values: dict[str, Any] | None = None,
    token: str | None = None,
    registry: BrickRegistry | None = None,
) -> tuple[RunResult, Report | None]:
    """Convenience wrapper: run, then re-validate the (possibly repaired) spec."""
    result = run_spec(spec, values=values, token=token, registry=registry)
    from daggrstudio.validator import validate

    report = validate(spec, registry, live=False) if result.steps else None
    return result, report