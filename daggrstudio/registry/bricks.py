"""Brick registry - the catalogue of verified, reusable pipeline steps.

Two layers, merged by id (verified layer wins):

* ``seed_spaces.json``   - hand-curated baseline shipped in the repo, from the
                           ``daggr`` / ``daggr-pipelines`` skills' tested-Space tables.
                           Entries carry ``"seed": true`` and must be re-verified before
                           being trusted for anything expensive.
* ``spaces.json``        - output of ``scripts/verify_registry.py``: live-introspected
                           endpoint names, parameter names and licenses.

Nothing in the app may invent a Space id, endpoint or parameter name: if a brick is not
in here, it does not exist as far as the planner, validator and medic are concerned.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent
SEED_FILE = DATA_DIR / "seed_spaces.json"
VERIFIED_FILE = DATA_DIR / "spaces.json"

#: Licenses that permit commercial use of the *output*. Anything not listed here is
#: treated as non-commercial until a human says otherwise.
COMMERCIAL_OK_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc0-1.0",
    "cc-by-4.0",
    "openrail",
    "openrail++",
    "openrail-m",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
    "mpl-2.0",
    "unlicense",
    "wtfpl",
    "llama3",
    "llama3.1",
    "llama3.2",
    "gemma",
    "qwen",
    "deepseek",
    "other",
}

#: License tokens that unambiguously forbid commercial use.
NON_COMMERCIAL_TOKENS = ("nc", "nd")
#: License substrings that unambiguously forbid commercial use.
NON_COMMERCIAL_MARKERS = ("non-commercial", "noncommercial", "research-only")


def commercial_ok(license_str: str | None) -> bool:
    """
    Decide commercial usability from the license string alone.

    Deliberately conservative and independent of the registry's ``commercial_ok`` flag:
    if a hand-written flag ever disagrees with the license text, the text wins.
    """
    if not license_str:
        return False
    lic = license_str.strip().lower()
    if lic in ("", "unknown", "other", "other-license", "no-license", "none"):
        return False
    tokens = [t for t in re.split(r"[^a-z0-9]+", lic) if t]
    if any(tok in tokens for tok in NON_COMMERCIAL_TOKENS):
        return False
    if any(marker in lic for marker in NON_COMMERCIAL_MARKERS):
        return False
    return True


@dataclass
class Brick:
    id: str
    kind: str = "space"  # space | inference_model
    source: str = ""
    api_name: str | None = None
    modality: str = "utility"
    industries: list[str] = field(default_factory=list)
    license: str = "unknown"
    commercial_ok: bool = False
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    output_kind: str = "json"
    postprocess_hint: str | None = None
    status: str = "unknown"  # running | sleeping | error | unknown
    runtime_error: str | None = None
    verified_at: str | None = None
    notes: str = ""
    seed: bool = False
    likes: int | None = None
    last_modified: str | None = None
    #: execution-probe outcome from scripts/verify_registry.py --probe
    #: ("ok" == we made a real call and it produced output; "" == never probed)
    probe: str = ""
    probe_seconds: float | None = None

    @property
    def is_live_verified(self) -> bool:
        return bool(self.verified_at) and self.status in ("running", "sleeping")

    @property
    def is_proven(self) -> bool:
        """True only when we have actually EXECUTED this brick successfully."""
        return self.probe == "ok"

    @property
    def commercial(self) -> bool:
        return self.commercial_ok and commercial_ok(self.license)

    @property
    def label(self) -> str:
        return f"{self.source}{self.api_name or ''}"

    def input_names(self) -> list[str]:
        return list(self.inputs.keys())

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "api_name": self.api_name,
            "modality": self.modality,
            "industries": list(self.industries),
            "license": self.license,
            "commercial_ok": self.commercial_ok,
            "inputs": dict(self.inputs),
            "outputs": list(self.outputs),
            "output_kind": self.output_kind,
            "postprocess_hint": self.postprocess_hint,
            "status": self.status,
            "runtime_error": self.runtime_error,
            "verified_at": self.verified_at,
            "notes": self.notes,
            "seed": self.seed,
        }
        if self.likes is not None:
            d["likes"] = self.likes
        if self.last_modified:
            d["last_modified"] = self.last_modified
        if self.probe:
            d["probe"] = self.probe
        if self.probe_seconds is not None:
            d["probe_seconds"] = self.probe_seconds
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Brick":
        return cls(
            id=str(d.get("id") or ""),
            kind=str(d.get("kind", "space")),
            source=str(d.get("source", "")),
            api_name=d.get("api_name"),
            modality=str(d.get("modality", "utility")),
            industries=list(d.get("industries") or []),
            license=str(d.get("license", "unknown")),
            commercial_ok=bool(d.get("commercial_ok", False)),
            inputs=dict(d.get("inputs") or {}),
            outputs=list(d.get("outputs") or []),
            output_kind=str(d.get("output_kind", "json")),
            postprocess_hint=d.get("postprocess_hint"),
            status=str(d.get("status", "unknown")),
            runtime_error=d.get("runtime_error"),
            verified_at=d.get("verified_at"),
            notes=str(d.get("notes", "")),
            seed=bool(d.get("seed", False)),
            likes=d.get("likes"),
            last_modified=d.get("last_modified"),
            probe=str(d.get("probe", "") or ""),
            probe_seconds=d.get("probe_seconds"),
        )

    # ── prompt/runtime helpers ──────────────────────────────────────────────────

    def signature_line(self) -> str:
        """One-line, LLM-facing description of the brick's exact call shape."""
        params = ", ".join(f"{n}: {t}" for n, t in self.inputs.items()) or "-"
        outs = ", ".join(self.outputs[:2]) or self.output_kind
        pp = f" postprocess={self.postprocess_hint}" if self.postprocess_hint else ""
        return (
            f"{self.id} | {self.source}{self.api_name or ''} | {self.modality} | "
            f"in({params}) -> {self.output_kind}{pp} | license={self.license} | "
            f"commercial={'yes' if self.commercial else 'NO'} | {self.status}"
            f"{' | proven' if self.is_proven else ''} | "
            f"industries={','.join(self.industries) or 'general'}"
        )


class BrickRegistry:
    """In-memory brick catalogue with graceful degradation to the seed layer."""

    def __init__(self, bricks: dict[str, Brick] | None = None, meta: dict[str, Any] | None = None):
        self._bricks: dict[str, Brick] = bricks or {}
        self.meta: dict[str, Any] = meta or {}

    # ── construction ────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, extra_paths: list[str | Path] | None = None) -> "BrickRegistry":
        bricks: dict[str, Brick] = {}
        meta: dict[str, Any] = {"sources": [], "loaded_at": datetime.now(timezone.utc).isoformat()}

        paths: list[Path] = [SEED_FILE, VERIFIED_FILE]
        for extra in extra_paths or []:
            paths.append(Path(extra))
        env_extra = os.environ.get("DAGGRSTUDIO_REGISTRY")
        if env_extra:
            paths.append(Path(env_extra))

        for path in paths:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except Exception as exc:  # a corrupt registry must not kill the app
                meta["sources"].append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            entries = payload.get("bricks") if isinstance(payload, dict) else payload
            count = 0
            for entry in entries or []:
                brick = Brick.from_dict(entry)
                if not brick.id:
                    continue
                previous = bricks.get(brick.id)
                # verified entries override seed entries; otherwise last file wins
                if previous and previous.is_live_verified and not brick.verified_at:
                    continue
                bricks[brick.id] = brick
                count += 1
            meta["sources"].append(
                {
                    "path": str(path),
                    "bricks": count,
                    "generated_at": (payload or {}).get("generated_at")
                    if isinstance(payload, dict)
                    else None,
                }
            )
        meta["total"] = len(bricks)
        return cls(bricks, meta)

    # ── queries ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._bricks)

    def __iter__(self):
        return iter(self._bricks.values())

    def all(self) -> list[Brick]:
        return list(self._bricks.values())

    def by_id(self, brick_id: str | None) -> Brick | None:
        if not brick_id:
            return None
        return self._bricks.get(brick_id)

    def by_source(self, source: str, api_name: str | None = None) -> Brick | None:
        for brick in self._bricks.values():
            if brick.source == source and (api_name is None or brick.api_name == api_name):
                return brick
        return None

    def find(
        self,
        modalities: list[str] | None = None,
        industries: list[str] | None = None,
        commercial_only: bool = False,
        exclude_ids: tuple[str, ...] = (),
        output_kind: str | None = None,
        include_not_running: bool = True,
        limit: int | None = None,
    ) -> list[Brick]:
        out = []
        for brick in self._bricks.values():
            if brick.id in exclude_ids:
                continue
            if modalities and brick.modality not in modalities:
                continue
            if industries and not set(industries) & set(brick.industries or ["general"]):
                continue
            if commercial_only and not brick.commercial:
                continue
            if output_kind and brick.output_kind != output_kind:
                continue
            if not include_not_running and brick.status not in ("running", "sleeping", "unknown"):
                continue
            out.append(brick)
        out.sort(key=lambda b: (b.status != "running", not b.is_proven,
                                not b.commercial, b.id))
        return out[:limit] if limit else out

    def alternatives(self, brick: Brick, commercial_only: bool = False) -> list[Brick]:
        """Replacements for a broken brick: same output kind first, then same modality."""
        cands = [
            b
            for b in self.find(modalities=[brick.modality], commercial_only=commercial_only)
            if b.id != brick.id
        ]
        same_kind = [b for b in cands if b.output_kind == brick.output_kind]
        rest = [b for b in cands if b.output_kind != brick.output_kind]
        return same_kind + rest

    def industries_present(self) -> list[str]:
        found: set[str] = set()
        for brick in self._bricks.values():
            found.update(brick.industries)
        return sorted(found or {"general"})

    def modalities_present(self) -> list[str]:
        return sorted({b.modality for b in self._bricks.values()})

    def catalogue_text(self, bricks: list[Brick], max_bricks: int = 40) -> str:
        """Compact, LLM-facing catalogue. Keeps prompts small and grounded."""
        lines = ["id | source | modality | signature | licence | commercial | status | industries"]
        for brick in bricks[:max_bricks]:
            lines.append(brick.signature_line())
        return "\n".join(lines)

    def summary(self) -> str:
        running = sum(1 for b in self._bricks.values() if b.status == "running")
        commercial = sum(1 for b in self._bricks.values() if b.commercial)
        proven = sum(1 for b in self._bricks.values() if b.is_proven)
        return (
            f"{len(self)} bricks | {running} reachable | {proven} proven to execute | "
            f"{commercial} commercially usable"
        )


_REGISTRY: BrickRegistry | None = None


def get_registry(refresh: bool = False, extra_paths: list[str | Path] | None = None) -> BrickRegistry:
    """Process-wide registry singleton (the JSON files are read once per process)."""
    global _REGISTRY
    if _REGISTRY is None or refresh:
        _REGISTRY = BrickRegistry.load(extra_paths)
    return _REGISTRY
