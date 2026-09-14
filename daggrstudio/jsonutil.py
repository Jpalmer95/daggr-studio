"""
Tolerant JSON extraction for LLM output.

Small/fast models wrap JSON in prose, markdown fences, or emit trailing commas. We never
trust the model's formatting: we locate the outermost JSON object and repair common
breakage instead of raising, so a slightly sloppy generation is not a hard failure.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|python|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json_text(text: str) -> str | None:
    """Return the most likely JSON object/array substring of `text`, or None."""
    if not text:
        return None
    for fence in _FENCE.findall(text):
        candidate = _first_json_block(fence)
        if candidate:
            return candidate
    return _first_json_block(text)


def _first_json_block(text: str) -> str | None:
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _repair(text: str) -> str:
    """Best-effort fixes for the failure modes we actually see from small models."""
    fixed = text.strip()
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')  # smart quotes
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    # single-quoted keys/strings → double quotes (only when no double quotes present)
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')
    return fixed


def parse_json(text: str) -> Any | None:
    """Parse JSON out of arbitrary model output; None when unparseable."""
    candidate = extract_json_text(text)
    if candidate is None:
        return None
    for attempt in (candidate, _repair(candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    # last resort: python literals (True/False/None)
    try:
        import ast

        return ast.literal_eval(_repair(candidate))
    except Exception:
        return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    data = parse_json(text)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
    return None


def dumps(obj: Any, *, indent: int | None = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
