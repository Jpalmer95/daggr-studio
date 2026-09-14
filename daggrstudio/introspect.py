"""
Live Space introspection, cached.

``gradio_client.Client(space).view_api()`` tells us the truth about an endpoint: its real
name, its real parameter names, their types, and which have defaults. That is what makes
validation real rather than guessed - and what lets the Medic notice that a sister Space
renamed a parameter under us.

Results are cached on disk (positive: 6h, negative: 10min) because introspection costs a
network round trip per brick and Spaces are slow to wake. Every failure is returned as
data, never raised: an unreachable Space is a *finding*, not a crash.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CACHE_TTL_OK = int(os.environ.get("DAGGRSTUDIO_INTROSPECT_TTL", 6 * 3600))
CACHE_TTL_FAIL = int(os.environ.get("DAGGRSTUDIO_INTROSPECT_FAIL_TTL", 600))
CACHE_PATH = Path(
    os.environ.get("DAGGRSTUDIO_CACHE", Path.home() / ".cache" / "daggr-studio" / "introspect.json")
)

#: Params whose name or type implies they take a file path.
FILE_PARAM_HINTS = ("image", "img", "photo", "audio", "video", "file", "mask", "media", "start_image")
SCALAR_TYPES = ("float", "int", "bool", "number")


@dataclass
class Param:
    name: str
    type: str = "Any"
    has_default: bool = False
    default: Any = None
    label: str = ""

    @property
    def is_file(self) -> bool:
        t = (self.type or "").lower()
        if "path" in t or "filedata" in t or t in ("filepath", "file"):
            return True
        return any(hint in self.name.lower() for hint in FILE_PARAM_HINTS)

    @property
    def is_scalar(self) -> bool:
        return any(s in (self.type or "").lower() for s in SCALAR_TYPES) and not self.is_file

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "has_default": self.has_default,
            "default": self.default,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Param":
        return cls(
            name=d.get("name", ""),
            type=d.get("type", "Any"),
            has_default=bool(d.get("has_default")),
            default=d.get("default"),
            label=d.get("label", ""),
        )


@dataclass
class SpaceInfo:
    """The live shape of one Space endpoint (or one provider model)."""

    source: str
    api_name: str | None = None
    ok: bool = False
    error: str | None = None
    error_class: str | None = None
    endpoints: dict[str, list[Param]] = field(default_factory=dict)
    return_types: dict[str, list[str]] = field(default_factory=dict)
    fetched_at: float = 0.0

    @property
    def ttl(self) -> int:
        return CACHE_TTL_OK if self.ok else CACHE_TTL_FAIL

    @property
    def params(self) -> list[Param]:
        return self.endpoints.get(self.api_name or "", [])

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def returns(self) -> list[str]:
        return self.return_types.get(self.api_name or "", [])

    @property
    def needs_token(self) -> bool:
        err = (self.error or "").lower()
        return "zerogpu" in err or "quota" in err or "gpu task" in err

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "api_name": self.api_name,
            "ok": self.ok,
            "error": self.error,
            "error_class": self.error_class,
            "endpoints": {k: [p.to_dict() for p in v] for k, v in self.endpoints.items()},
            "return_types": self.return_types,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpaceInfo":
        return cls(
            source=d.get("source", ""),
            api_name=d.get("api_name"),
            ok=bool(d.get("ok")),
            error=d.get("error"),
            error_class=d.get("error_class"),
            endpoints={
                k: [Param.from_dict(p) for p in v] for k, v in (d.get("endpoints") or {}).items()
            },
            return_types=dict(d.get("return_types") or {}),
            fetched_at=float(d.get("fetched_at") or 0),
        )


# ─── disk cache ───────────────────────────────────────────────────────────────


def _read_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _write_cache(payload: dict[str, Any]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload))
    except Exception:
        pass  # a read-only FS must not break introspection


def _cache_key(source: str, api_name: str | None) -> str:
    return f"{source}::{api_name or ''}"


# ─── introspection ────────────────────────────────────────────────────────────


def _summarise_type(t: Any) -> str:
    """
    Normalise gradio_client's very verbose type strings into a small vocabulary:
    file | str | number | bool | list | dict | tuple | image | audio | video | Any.

    Being lossy here is fine *because* the vocabulary is what the validator and the
    postprocess-hint logic reason about - but tuple-ness must survive, since a multi-value
    return is exactly what breaks naive pipelines.
    """
    text = str(t or "Any").lower()
    head = text.split("(")[0].strip()
    if "path" in text or "filedata" in text or "filepath" in text:
        return "file"
    if head.startswith("tuple"):
        return "tuple"
    if "list[" in text or head.startswith("list"):
        return "list"
    if "str" in text:
        return "str"
    if "bool" in text:
        return "bool"
    if any(k in text for k in ("float", "int", "number")):
        return "number"
    if head.startswith("dict"):
        return "dict"
    return head[:40] or "Any"


def expand_returns(returns: list[str]) -> list[str]:
    """
    Expand a Gradio return list so each *element* of a tuple return gets its own entry.

    ``tuple[FileData, FileData]`` becomes ``["file", "file"]``. This is what lets the
    registry record "this Space returns (original, processed)" - the difference between a
    working background-removal step and one that silently returns the wrong image.
    """
    out: list[str] = []
    for raw in returns or []:
        kind = _summarise_type(raw)
        if kind != "tuple":
            out.append(kind)
            continue
        elements = _split_tuple_elements(str(raw))
        out.extend(_summarise_type(e) for e in elements) if elements else out.append("tuple")
    return out


def _split_tuple_elements(text: str) -> list[str]:
    """Split the top level of a ``tuple[...]`` type string into its element type strings."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return []
    inner = text[start + 1 : end]
    parts, depth, current = [], 0, []
    for ch in inner:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def introspect(
    source: str,
    api_name: str | None = None,
    force: bool = False,
    token: str | None = None,
    timeout_s: int = 60,
) -> SpaceInfo:
    """
    Return the live endpoint map of ``source``, from cache when fresh.

    Never raises: failures come back as ``SpaceInfo(ok=False, error=...)``.
    """
    key = _cache_key(source, api_name)
    cache = _read_cache()
    entry = cache.get(key)
    if entry and not force:
        info = SpaceInfo.from_dict(entry)
        if time.time() - info.fetched_at < info.ttl:
            return info

    info = _fetch(source, api_name, token, timeout_s)
    cache[key] = info.to_dict()
    _write_cache(cache)
    return info


def _fetch(source: str, api_name: str | None, token: str | None, timeout_s: int) -> SpaceInfo:
    info = SpaceInfo(source=source, api_name=api_name, fetched_at=time.time())
    try:
        from gradio_client import Client

        token = token or os.environ.get("HF_TOKEN") or None
        client = Client(source, token=token, verbose=False) if token else Client(source, verbose=False)
        api = client.view_api(return_format="dict", print_info=False)
        named = api.get("named_endpoints") or {}
        for name, ep in named.items():
            if not isinstance(ep, dict):
                continue
            params = []
            for p in ep.get("parameters", []) or []:
                params.append(
                    Param(
                        name=p.get("parameter_name") or "",
                        type=_summarise_type((p.get("python_type") or {}).get("type")),
                        has_default=bool(p.get("parameter_has_default")),
                        default=p.get("parameter_default"),
                        label=p.get("label") or "",
                    )
                )
            info.endpoints[name] = params
            info.return_types[name] = expand_returns(
                [(r.get("python_type") or {}).get("type") for r in (ep.get("returns") or [])]
            )
        info.ok = bool(named)
        if not info.ok:
            info.error = "Space exposes no named API endpoints (Use via API is unavailable)"
            info.error_class = "NoNamedEndpoints"
        elif api_name and api_name not in named:
            info.error = (
                f"endpoint '{api_name}' not found; available: {sorted(named)}"
            )
            info.error_class = "EndpointMissing"
    except Exception as exc:
        info.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        info.error_class = type(exc).__name__
    return info


def cached_info(source: str, api_name: str | None = None) -> SpaceInfo | None:
    """Cache-only lookup (no network); used where a run must not block on the Hub."""
    entry = _read_cache().get(_cache_key(source, api_name))
    return SpaceInfo.from_dict(entry) if entry else None


def clear_cache() -> int:
    cache = _read_cache()
    n = len(cache)
    _write_cache({})
    return n


def cache_stats() -> dict[str, Any]:
    cache = _read_cache()
    ok = sum(1 for v in cache.values() if v.get("ok"))
    return {"entries": len(cache), "ok": ok, "failed": len(cache) - ok, "path": str(CACHE_PATH)}