"""Registry: licence policy and brick selection are safety-critical, so they are tested."""

from __future__ import annotations

import json

from daggrstudio.registry.bricks import Brick, BrickRegistry, commercial_ok, get_registry


def test_seed_registry_loads_and_is_usable():
    reg = get_registry()
    assert len(reg) >= 15
    assert reg.by_id("flux1-schnell") is not None
    summary = reg.summary()
    assert "bricks" in summary and "commercially usable" in summary


def test_registry_overlay_prefers_verified_layer(tmp_path):
    verified = tmp_path / "spaces.json"
    verified.write_text(json.dumps({"bricks": [
        {"id": "flux1-schnell", "kind": "space", "source": "black-forest-labs/FLUX.1-schnell",
         "api_name": "/infer", "modality": "image-gen", "license": "apache-2.0",
         "commercial_ok": True, "inputs": {"prompt": "str"}, "outputs": ["filepath"],
         "output_kind": "image", "status": "running", "verified_at": "2026-09-14T00:00:00Z",
         "postprocess_hint": "tuple_index:0", "industries": ["art"]},
    ]}))
    reg = BrickRegistry.load(extra_paths=[verified])
    brick = reg.by_id("flux1-schnell")
    assert brick.verified_at == "2026-09-14T00:00:00Z"  # verified wins over seed
    assert brick.is_live_verified


def test_corrupt_registry_file_does_not_crash_the_app(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json at all")
    reg = BrickRegistry.load(extra_paths=[bad])
    assert len(reg) >= 15  # seed still loaded
    assert any("error" in s for s in reg.meta["sources"])


def test_commercial_licence_policy():
    assert commercial_ok("apache-2.0") is True
    assert commercial_ok("mit") is True
    assert commercial_ok("cc-by-4.0") is True
    assert commercial_ok("cc-by-nc-4.0") is False
    assert commercial_ok("cc-by-nd-4.0") is False
    assert commercial_ok("non-commercial") is False
    assert commercial_ok("unknown") is False
    assert commercial_ok("") is False
    assert commercial_ok(None) is False


def test_brick_commercial_requires_flag_and_licence_to_agree():
    ok = Brick(id="a", license="apache-2.0", commercial_ok=True)
    assert ok.commercial is True
    # flag claims commercial but the licence text says otherwise -> text wins
    liar = Brick(id="b", license="cc-by-nc-4.0", commercial_ok=True)
    assert liar.commercial is False


def test_find_filters_by_modality_industry_and_licence():
    reg = get_registry()
    image = reg.find(modalities=["image-gen"])
    assert image and all(b.modality == "image-gen" for b in image)

    commercial = reg.find(modalities=["image-gen"], commercial_only=True)
    assert all(b.commercial for b in commercial)

    game = reg.find(industries=["game-dev"])
    assert game and all("game-dev" in (b.industries or ["general"]) for b in game)

    assert reg.find(modalities=["video"], commercial_only=True, exclude_ids=("ltx2-distilled",))


def test_find_excludes_requested_ids():
    reg = get_registry()
    ids = {b.id for b in reg.find(modalities=["image-gen"], exclude_ids=("flux1-schnell",))}
    assert "flux1-schnell" not in ids


def test_alternatives_prefers_same_output_kind():
    reg = get_registry()
    broken = reg.by_id("flux1-schnell")
    alts = reg.alternatives(broken)
    assert alts and broken.id not in {b.id for b in alts}
    assert alts[0].output_kind == broken.output_kind


def test_running_bricks_sort_first():
    reg = get_registry()
    found = reg.find(modalities=["image-gen"])
    statuses = [b.status == "running" for b in found]
    assert statuses == sorted(statuses, reverse=True)


def test_signature_line_is_grounded_and_compact():
    reg = get_registry()
    line = reg.catalogue_text([reg.by_id("background-removal")])
    assert "hf-applications/background-removal" in line
    assert "/image" in line
    assert "license=" in line  # whatever the Hub currently reports, it must be shown


def test_by_source_lookup():
    reg = get_registry()
    brick = reg.by_source("black-forest-labs/FLUX.1-schnell", "/infer")
    assert brick is not None and brick.id == "flux1-schnell"
    assert reg.by_source("nobody/nothing") is None