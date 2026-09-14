"""
Deterministic code generation: WorkflowSpec -> live daggr Graph, and WorkflowSpec -> app.py.

Two entry points, one source of truth:

* :func:`build_graph`   — the object the app actually executes in-process.
* :func:`render_app_py` — a standalone, readable daggr script the user can download,
  run locally (`daggr app.py`) or deploy as their own Space.

Because both are driven by the same spec, "the code I downloaded" and "the workflow I ran
in the Space" cannot drift apart. No LLM touches this module.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from daggrstudio.codegen.fn_library import FN_LIBRARY
from daggrstudio.registry.bricks import Brick, BrickRegistry, get_registry
from daggrstudio.spec import Binding, InputPort, Step, WorkflowSpec

# ─── hidden value callables ───────────────────────────────────────────────────
# Specs may only reference these names (enforced by WorkflowSpec.validate_shape).

STYLE_SUFFIXES = [
    "highly detailed",
    "clean vector style",
    "cinematic lighting",
    "flat colour, game-ready",
]


def _random_int() -> int:
    return random.randint(0, 2**31 - 1)


def _random_float() -> float:
    return random.random()


def _uuid4() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _random_choice_style() -> str:
    return random.choice(STYLE_SUFFIXES)


CALLABLE_FNS: dict[str, Callable[[], Any]] = {
    "random_int": _random_int,
    "random_float": _random_float,
    "uuid4": _uuid4,
    "now_iso": _now_iso,
    "random_choice_style": _random_choice_style,
}


# ─── postprocess hints ────────────────────────────────────────────────────────


def resolve_postprocess(hint: str | None) -> Callable[..., Any] | None:
    """
    Turn a portable hint string into a callable daggr can use.

    Hints (not lambdas) are what live in the spec, so they survive JSON round-trips and
    can be written to a shared Dataset.
      * ``tuple_index:N`` / ``first`` — keep element N of a multi-value return
      * ``dict_path:<key>``           — keep one key of a returned dict
      * ``unwrap_path``               — normalise {"path": ...} / FileData to a path
    """
    if not hint:
        return None
    hint = hint.strip()

    if hint in ("first", "keep_first"):
        hint = "tuple_index:0"

    if hint.startswith("tuple_index:"):
        try:
            idx = int(hint.split(":", 1)[1])
        except ValueError:
            return None
        return lambda *args: args[idx] if len(args) > idx else (args[0] if args else None)

    if hint.startswith("dict_path:"):
        key = hint.split(":", 1)[1]

        def _dict_path(result: Any, *_: Any) -> Any:
            if isinstance(result, dict):
                return result.get(key)
            return result

        return _dict_path

    if hint == "unwrap_path":

        def _unwrap_path(result: Any, *_: Any) -> Any:
            if isinstance(result, dict):
                return result.get("path", result)
            return result

        return _unwrap_path

    return None


def infer_postprocess_hint(brick: Brick | None, step: Step) -> str | None:
    """Registry hint wins; otherwise derive one from the declared return arity."""
    if step.postprocess:
        return step.postprocess
    if brick and brick.postprocess_hint:
        return brick.postprocess_hint
    if len(step.outputs) == 1 and brick and len(brick.outputs) > 1:
        return "tuple_index:0"
    return None


# ─── gradio components ────────────────────────────────────────────────────────


def make_input_component(port: InputPort):
    import gradio as gr

    label = port.label or port.port.replace("_", " ").title()
    comp = port.component
    if comp == "textbox":
        return gr.Textbox(label=label, value=port.default if port.default is not None else "",
                          lines=port.lines or 3)
    if comp == "image":
        return gr.Image(label=label)
    if comp == "audio":
        return gr.Audio(label=label)
    if comp == "video":
        return gr.Video(label=label)
    if comp == "model3d":
        return gr.Model3D(label=label)
    if comp == "number":
        return gr.Number(label=label, value=port.default if port.default not in ("", None) else 0)
    if comp == "slider":
        return gr.Slider(
            minimum=port.minimum if port.minimum is not None else 0,
            maximum=port.maximum if port.maximum is not None else 100,
            value=port.default if port.default not in ("", None) else 0,
            step=port.step or 1,
            label=label,
        )
    if comp == "dropdown":
        choices = port.choices or ["auto"]
        return gr.Dropdown(choices=choices, value=port.default or choices[0], label=label)
    if comp == "checkbox":
        return gr.Checkbox(label=label, value=bool(port.default))
    if comp == "json":
        return gr.JSON(label=label)
    if comp == "gallery":
        return gr.Gallery(label=label)
    if comp == "file":
        return gr.File(label=label)
    return gr.Textbox(label=label)


def make_output_component(spec_out: dict[str, Any]):
    import gradio as gr

    label = spec_out.get("label") or ""
    comp = (spec_out.get("component") or "textbox").lower()
    if comp == "image":
        return gr.Image(label=label)
    if comp == "audio":
        return gr.Audio(label=label)
    if comp == "video":
        return gr.Video(label=label)
    if comp == "model3d":
        return gr.Model3D(label=label)
    if comp == "json":
        return gr.JSON(label=label)
    if comp == "gallery":
        return gr.Gallery(label=label)
    if comp == "file":
        return gr.File(label=label)
    if comp == "number":
        return gr.Number(label=label)
    return gr.Textbox(label=label, lines=6)


# ─── spec -> live Graph ───────────────────────────────────────────────────────


def resolve_brick(step: Step, registry: BrickRegistry) -> Brick | None:
    """Registry is authoritative for source/api_name; the spec is only a pointer."""
    brick = registry.by_id(step.brick_id)
    if brick is None and step.source:
        brick = registry.by_source(step.source, step.api_name)
    return brick


def effective_source(step: Step, brick: Brick | None) -> tuple[str, str | None]:
    if brick:
        return brick.source, brick.api_name
    return step.source or "", step.api_name


def build_binding(
    binding: Binding,
    inputs_node: Any,
    built_nodes: dict[str, Any],
) -> Any:
    """Translate a spec binding into the concrete object daggr expects."""
    if binding.kind == "user":
        return getattr(inputs_node, binding.user_port)
    ref = binding.step_ref
    if binding.kind == "step" and ref:
        upstream, port = ref
        if upstream not in built_nodes:
            raise KeyError(f"edge references step '{upstream}' which has not been built")
        return getattr(built_nodes[upstream], port)
    if binding.kind == "callable":
        fn = CALLABLE_FNS.get(binding.callable or "")
        if fn is None:
            raise KeyError(f"unknown callable '{binding.callable}'")
        return fn
    return binding.value


def node_name_for(step: Step, used: set[str] | None = None) -> str | None:
    """
    URL-safe daggr node name for a step.

    daggr uses the node name as a dict key and in API paths (``/api/run/{node_name}``),
    so it must be unique, ASCII and space-free. ``fn`` steps pass ``None`` because daggr
    already names them after the function (which is readable).
    """
    import re

    if step.kind == "fn":
        return None
    base = re.sub(r"[^a-z0-9_]+", "_", (step.id or "step").lower()).strip("_") or "step"
    if used is None:
        return base
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def build_graph(
    spec: WorkflowSpec,
    registry: BrickRegistry | None = None,
    persist_key: str | None = None,
) -> Any:
    """
    Build a live ``daggr.Graph`` from a spec.

    Raises KeyError/ValueError on structural problems rather than producing a half-built
    graph; run :func:`daggrstudio.validator.validate` first and heal, then call this.
    """
    from daggr import FnNode, GradioNode, InferenceNode, InputNode, Graph

    registry = registry or get_registry()
    built_nodes: dict[str, Any] = {}
    daggr_nodes: list[Any] = []
    used_names: set[str] = set()

    inputs_node = None
    if spec.inputs:
        inputs_node = InputNode(
            name="Inputs",
            ports={p.port: make_input_component(p) for p in spec.inputs},
        )
        daggr_nodes.append(inputs_node)

    for step in spec.steps:
        brick = resolve_brick(step, registry)
        if step.kind in ("space", "inference") and brick is None:
            # Fail fast with an actionable message instead of letting daggr attempt a
            # network lookup for a brick that does not exist.
            raise KeyError(
                f"step '{step.id}' references unknown brick "
                f"{step.brick_id or step.source!r}; it is not in the registry"
            )
        node_inputs = {
            port: build_binding(binding, inputs_node, built_nodes)
            for port, binding in step.inputs.items()
        }
        node_outputs = {
            port: make_output_component(meta) for port, meta in step.outputs.items()
        }
        display_name = node_name_for(step, used_names)

        if step.kind == "fn":
            fn = FN_LIBRARY.get(step.fn or "")
            if fn is None:
                raise KeyError(f"step '{step.id}': unknown fn '{step.fn}'")
            node = FnNode(
                fn,
                name=display_name,
                inputs=node_inputs,
                outputs=node_outputs,
                concurrent=bool(step.concurrent),
                concurrency_group=step.concurrency_group,
            )
        elif step.kind == "inference":
            source, _ = effective_source(step, brick)
            node = InferenceNode(
                source,
                name=display_name,
                inputs=node_inputs,
                outputs=node_outputs,
                postprocess=resolve_postprocess(infer_postprocess_hint(brick, step)),
            )
        else:
            source, api_name = effective_source(step, brick)
            node = GradioNode(
                source,
                api_name=api_name,
                name=display_name,
                inputs=node_inputs,
                outputs=node_outputs,
                postprocess=resolve_postprocess(infer_postprocess_hint(brick, step)),
            )

        built_nodes[step.id] = node
        daggr_nodes.append(node)

    if not daggr_nodes:
        raise ValueError("spec has no nodes to build")

    return Graph(
        name=spec.name,
        nodes=daggr_nodes,
        persist_key=persist_key or f"daggr-studio-{spec.slug}",
    )


# ─── spec -> readable app.py ──────────────────────────────────────────────────


def _py(value: Any) -> str:
    """Python literal for a JSON value."""
    import json

    return json.dumps(value, ensure_ascii=False)


def _binding_expr(binding: Binding, var_of: dict[str, str]) -> str:
    if binding.kind == "user":
        return f"inputs.{binding.user_port}"
    ref = binding.step_ref
    if binding.kind == "step" and ref:
        upstream, port = ref
        var = var_of.get(upstream, upstream)
        return f"{var}.{port}"
    if binding.kind == "callable":
        return CALLABLE_SOURCE.get(binding.callable or "", "lambda: None")
    return _py(binding.value)


#: Source snippets emitted for hidden value callables, so the downloaded app.py has no
#: dependency on this package.
CALLABLE_SOURCE = {
    "random_int": "lambda: random.randint(0, 2**31 - 1)",
    "random_float": "random.random",
    "uuid4": "lambda: str(uuid.uuid4())",
    "now_iso": "lambda: datetime.now(timezone.utc).isoformat(timespec='seconds')",
    "random_choice_style": "lambda: random.choice(STYLE_SUFFIXES)",
}


def _postprocess_source(hint: str | None) -> str | None:
    if not hint:
        return None
    hint = hint.strip()
    if hint in ("first", "keep_first"):
        hint = "tuple_index:0"
    if hint.startswith("tuple_index:"):
        return f"lambda *args: args[{int(hint.split(':', 1)[1])}] if args else None"
    if hint.startswith("dict_path:"):
        key = hint.split(":", 1)[1]
        return f"lambda result, *_: (result.get({_py(key)}) if isinstance(result, dict) else result)"
    if hint == "unwrap_path":
        return "lambda result, *_: (result.get('path', result) if isinstance(result, dict) else result)"
    return None


def render_app_py(spec: WorkflowSpec, registry: BrickRegistry | None = None) -> str:
    """
    Render a standalone daggr app for this spec.

    The output is meant to be readable and self-contained: someone can open it, see how
    their workflow is wired, `pip install daggr`, run `daggr app.py`, or push it as a
    Space of their own.
    """
    registry = registry or get_registry()
    var_of = {step.id: f"step_{i}" for i, step in enumerate(spec.steps)}
    lines: list[str] = []
    add = lines.append

    add('"""')
    add(f"{spec.name}")
    add("")
    if spec.intent:
        add(f"Goal: {spec.intent}")
    add("")
    add("Generated by Daggr Studio (bricks -> daggr). Deterministic codegen: this file is")
    add("exactly what ran in the Space. Run it with:  pip install daggr && daggr app.py")
    add("")
    add(f"Bricks used: {', '.join(spec.brick_ids) or 'none'}")
    add(f"License posture: {spec.license_posture} | compute tier: {spec.compute_tier}")
    add(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    add('"""')
    add("")
    add("import random")
    add("import uuid")
    add("from datetime import datetime, timezone")
    add("from pathlib import Path")
    add("")
    add("import gradio as gr")
    add("from daggr import FnNode, GradioNode, InferenceNode, InputNode, Graph")
    add("")
    add("STYLE_SUFFIXES = " + _py(STYLE_SUFFIXES))
    add("")

    # local helper functions actually referenced by the spec
    used_fns = [s.fn for s in spec.steps if s.kind == "fn" and s.fn]
    if used_fns:
        add("# ── local helper functions ────────────────────────────────────────────────")
        for name in used_fns:
            add(_fn_source(name))
            add("")

    if spec.inputs:
        add("# ── inputs ──────────────────────────────────────────────────────────────")
        add("inputs = InputNode(")
        add('    name="Inputs",')
        add("    ports={")
        for port in spec.inputs:
            add(f"        {_py(port.port)}: {_input_component_source(port)},")
        add("    },")
        add(")")
        add("")

    add("# ── steps ────────────────────────────────────────────────────────────────")
    for step in spec.steps:
        brick = resolve_brick(step, registry)
        source, api_name = effective_source(step, brick)
        var = var_of[step.id]
        add(f"# {step.title or step.id}" + (f" — {step.why}" if step.why else ""))
        if brick:
            add(f"# brick: {brick.id} ({brick.source}) | license: {brick.license}"
                f" | commercial: {brick.commercial}")
        if step.kind == "fn":
            add(f"{var} = FnNode(")
            add(f"    {step.fn},")
            add(f'    name={_py(step.title or step.id)},')
            add(f"    inputs={{")
            for port, binding in step.inputs.items():
                add(f"        {_py(port)}: {_binding_expr(binding, var_of)},")
            add("    },")
            if step.outputs:
                add(f"    outputs={{")
                for port, meta in step.outputs.items():
                    add(f"        {_py(port)}: {_output_component_source(meta)},")
                add("    },")
            add(f"    concurrent={bool(step.concurrent)},")
            add(")")
        elif step.kind == "inference":
            add(f"{var} = InferenceNode(")
            add(f"    {_py(source)},")
            add(f'    name={_py(step.title or step.id)},')
            add(f"    inputs={{")
            for port, binding in step.inputs.items():
                add(f"        {_py(port)}: {_binding_expr(binding, var_of)},")
            add("    },")
            add(f"    outputs={{")
            for port, meta in step.outputs.items():
                add(f"        {_py(port)}: {_output_component_source(meta)},")
            add("    },")
            pp = _postprocess_source(infer_postprocess_hint(brick, step))
            if pp:
                add(f"    postprocess={pp},")
            add(")")
        else:
            add(f"{var} = GradioNode(")
            add(f"    {_py(source)},          # {brick.modality if brick else 'space'}")
            add(f"    api_name={_py(api_name)},")
            add(f'    name={_py(step.title or step.id)},')
            add(f"    inputs={{")
            for port, binding in step.inputs.items():
                add(f"        {_py(port)}: {_binding_expr(binding, var_of)},")
            add("    },")
            add(f"    outputs={{")
            for port, meta in step.outputs.items():
                add(f"        {_py(port)}: {_output_component_source(meta)},")
            add("    },")
            pp = _postprocess_source(infer_postprocess_hint(brick, step))
            if pp:
                add(f"    postprocess={pp},")
            add(")")
        add("")

    add("# ── graph ────────────────────────────────────────────────────────────────")
    add("graph = Graph(")
    add(f"    name={_py(spec.name)},")
    node_vars = (["inputs"] if spec.inputs else []) + [var_of[s.id] for s in spec.steps]
    add(f"    nodes=[{', '.join(node_vars)}],")
    add(f"    persist_key={_py('daggr-studio-' + spec.slug)},")
    add(")")
    add("")
    add('if __name__ == "__main__":')
    add("    graph.launch()")
    add("")
    return "\n".join(lines)


def _input_component_source(port: InputPort) -> str:
    label = _py(port.label or port.port.replace("_", " ").title())
    comp = port.component
    if comp == "textbox":
        default = _py(port.default if port.default is not None else "")
        return f"gr.Textbox(label={label}, value={default}, lines={port.lines or 3})"
    if comp == "image":
        return f"gr.Image(label={label})"
    if comp == "audio":
        return f"gr.Audio(label={label})"
    if comp == "video":
        return f"gr.Video(label={label})"
    if comp == "model3d":
        return f"gr.Model3D(label={label})"
    if comp == "number":
        return f"gr.Number(label={label}, value={port.default if port.default not in ('', None) else 0})"
    if comp == "slider":
        return (
            f"gr.Slider(minimum={port.minimum if port.minimum is not None else 0}, "
            f"maximum={port.maximum if port.maximum is not None else 100}, "
            f"value={port.default if port.default not in ('', None) else 0}, "
            f"step={port.step or 1}, label={label})"
        )
    if comp == "dropdown":
        choices = _py(port.choices or ["auto"])
        return f"gr.Dropdown(choices={choices}, value={_py(port.default) if port.default else None}, label={label})"
    if comp == "checkbox":
        return f"gr.Checkbox(label={label}, value={bool(port.default)})"
    if comp == "json":
        return f"gr.JSON(label={label})"
    if comp == "gallery":
        return f"gr.Gallery(label={label})"
    if comp == "file":
        return f"gr.File(label={label})"
    return f"gr.Textbox(label={label})"


def _output_component_source(meta: dict[str, Any]) -> str:
    label = _py(meta.get("label") or "")
    comp = (meta.get("component") or "textbox").lower()
    mapping = {
        "image": "gr.Image",
        "audio": "gr.Audio",
        "video": "gr.Video",
        "model3d": "gr.Model3D",
        "json": "gr.JSON",
        "gallery": "gr.Gallery",
        "file": "gr.File",
        "number": "gr.Number",
    }
    if comp in mapping:
        return f"{mapping[comp]}(label={label})"
    return f"gr.Textbox(label={label}, lines=6)"


def _fn_source(name: str) -> str:
    """Emit the source of a built-in fn so the generated app does not import this package."""
    import inspect

    from daggrstudio.codegen import fn_library

    fn = fn_library.FN_LIBRARY.get(name)
    if fn is None:
        return f"# unknown helper: {name}"
    src = inspect.getsource(fn)
    src = "\n".join(line for line in src.splitlines() if not line.strip().startswith("@"))
    return src.rstrip() + "\n"