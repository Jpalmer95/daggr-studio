"""
Built-in local functions available to ``kind="fn"`` steps.

Why a fixed library instead of arbitrary code? A spec is model-authored, so it must never
be able to smuggle in executable Python. The planner/medic may only reference names in
``FN_LIBRARY``; anything else is rejected by ``WorkflowSpec.validate_shape`` and by the
validator. Adding a capability = adding a function here (reviewed, tested).
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def join_texts(a: str, b: str = "") -> str:
    """Concatenate two text values (separator-aware)."""
    parts = [str(p).strip() for p in (a, b) if p is not None and str(p).strip()]
    return "\n\n".join(parts)


def text_to_lines(text: str) -> list[str]:
    """Split text into non-empty lines - useful for batch steps downstream."""
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def pick_first(value: Any) -> Any:
    """Pass through the first element when given a list/tuple, else the value itself."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def summarize_stats(payload: Any) -> dict[str, Any]:
    """Small JSON summary of any payload - keeps the canvas readable."""
    if isinstance(payload, dict):
        return {"kind": "dict", "keys": list(payload)[:20], "size": len(payload)}
    if isinstance(payload, (list, tuple)):
        return {"kind": type(payload).__name__, "length": len(payload)}
    text = str(payload)
    return {"kind": type(payload).__name__, "chars": len(text), "preview": text[:200]}


def json_report(payload: Any) -> dict[str, Any]:
    """Normalise any value into a JSON-serialisable report."""
    return {"payload": payload, "generated_at": datetime.now(timezone.utc).isoformat()}


def organize_outputs(source: Any, output_dir: str = "./output/workflow") -> dict[str, Any]:
    """
    Terminal step: copy generated artifact(s) into a timestamped folder with a manifest.

    Accepts a path, a list of paths, or a dict of name -> path (daggr hands file values
    around as path strings).
    """
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    target = Path(output_dir).expanduser() / stamp
    target.mkdir(parents=True, exist_ok=True)

    candidates: list[str] = []
    if isinstance(source, str):
        candidates = [source]
    elif isinstance(source, dict):
        candidates = [v for v in source.values() if isinstance(v, str)]
    elif isinstance(source, (list, tuple)):
        candidates = [v for v in source if isinstance(v, str)]

    copied: dict[str, str] = {}
    for path in candidates:
        if path and os.path.exists(path):
            dest = target / Path(path).name
            shutil.copy2(path, dest)
            copied[Path(path).name] = str(dest)

    manifest = {"output_dir": str(target), "files": copied, "count": len(copied)}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


#: name -> function. Keys are the only legal ``Step.fn`` values.
FN_LIBRARY: dict[str, Any] = {
    "join_texts": join_texts,
    "text_to_lines": text_to_lines,
    "pick_first": pick_first,
    "summarize_stats": summarize_stats,
    "json_report": json_report,
    "organize_outputs": organize_outputs,
}

FN_DESCRIPTIONS: dict[str, str] = {
    "join_texts": "Join two text values with a blank line (a, b='') -> str",
    "text_to_lines": "Split text into a list of non-empty lines (text) -> list[str]",
    "pick_first": "Take the first element of a list/tuple (value) -> Any",
    "summarize_stats": "Summarise any payload as a small JSON dict (payload) -> dict",
    "json_report": "Wrap any payload in a JSON report (payload) -> dict",
    "organize_outputs": "Copy artifacts to a timestamped folder + manifest (source, output_dir) -> dict",
}


def fn_signatures() -> dict[str, dict[str, str]]:
    """
    name -> {param_name: annotation} for every built-in fn.

    The validator uses this to catch specs that wire a parameter the function does not
    accept (a failure mode daggr would otherwise raise only at run time).
    """
    sigs: dict[str, dict[str, str]] = {}
    for name, fn in FN_LIBRARY.items():
        params: dict[str, str] = {}
        for pname, param in inspect.signature(fn).parameters.items():
            if pname in ("self", "cls"):
                continue
            ann = "Any" if param.annotation is inspect.Parameter.empty else str(param.annotation)
            params[pname] = ann.replace("<class '", "").replace("'>", "")
        sigs[name] = params
    return sigs


def fn_catalogue_text() -> str:
    return "\n".join(f"{name} | {desc}" for name, desc in FN_DESCRIPTIONS.items())
