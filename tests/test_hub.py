"""Sharing layer: leaderboard filtering, export/import, and the no-token degradation path.

No network here: the Hub paths are exercised against the local mirror and against stubs.
"""

from __future__ import annotations

import json

import pytest

from daggrstudio import hub
from daggrstudio.hub import (
    export_spec,
    import_spec,
    leaderboard,
    list_workflows,
    spec_from_entry,
    spec_metadata,
    stats,
)
from tests.conftest import concept_spec


@pytest.fixture(autouse=True)
def isolated_mirror(monkeypatch, tmp_path):
    monkeypatch.setattr(hub, "LOCAL_MIRROR", tmp_path / "workflows")
    yield


def _row(name: str, industry: str, posture: str, votes: int, bricks: list[str],
         tier: str = "cloud-free", published: str = "2026-09-14T00:00:00+00:00") -> dict:
    return {"slug": name.lower().replace(" ", "-"), "name": name, "industry": industry,
            "license_posture": posture, "votes": votes, "brick_ids": bricks, "steps": len(bricks),
            "compute_tier": tier, "published_at": published, "modalities": ["image-gen"],
            "tags": [industry], "author": "tester"}


def test_metadata_is_derived_not_hand_written():
    spec = concept_spec()
    meta = spec_metadata(spec, username="jkorstad")
    assert meta["slug"] == spec.slug
    assert meta["author"] == "jkorstad"
    assert meta["steps"] == 3
    assert meta["brick_ids"] == spec.brick_ids
    assert meta["modalities"]  # derived from the registry, not guessed
    assert meta["votes"] == 0 and meta["hf_user"] == "jkorstad"


def test_publishing_without_a_token_saves_locally_and_says_so():
    spec = concept_spec()
    result = hub.publish(spec, token=None)
    assert result.ok is False
    assert "saved to this Space only" in result.message
    # the file is still available for export
    assert (hub.LOCAL_MIRROR / f"{spec.slug}.json").exists()
    rows = list_workflows(token=None, include_mirror=True)
    assert any(r["slug"] == spec.slug for r in rows)


def test_list_workflows_merges_the_remote_index_with_local_saves(monkeypatch):
    monkeypatch.setattr(hub, "_read_remote_json",
                        lambda filename, token=None: {"workflows": {
                            "remote-flow": _row("Remote flow", "art", "commercial-only", 3,
                                                ["flux1-schnell"])}})
    hub.publish(concept_spec(), token=None)
    rows = {r["slug"] for r in list_workflows(token=None)}
    assert "remote-flow" in rows and "concept-to-sprite" in rows


def test_leaderboard_filters_by_modality_industry_and_licence():
    rows = [
        _row("Sprite maker", "game-dev", "commercial-only", 5, ["flux1-schnell"]),
        _row("NC experiment", "art", "any", 99, ["hunyuan3d-2"]),
        _row("Music thing", "music-production", "commercial-only", 2, ["musicgen"]),
    ]
    commercial = leaderboard(rows, license_filter="commercial-only")
    assert {r["name"] for r in commercial} == {"Sprite maker", "Music thing"}

    game = leaderboard(rows, industry="game-dev")
    assert [r["name"] for r in game] == ["Sprite maker"]

    nc = leaderboard(rows, license_filter="any", sort="votes")
    assert nc[0]["name"] == "NC experiment"  # 99 votes ranks first


def test_leaderboard_sorts_and_limits():
    rows = [_row(f"Flow {i}", "art", "commercial-only", i, ["flux1-schnell"]) for i in range(10)]
    top = leaderboard(rows, sort="votes", limit=3)
    assert [r["votes"] for r in top] == [9, 8, 7]
    recent = leaderboard(rows, sort="recent", limit=1)
    assert len(recent) == 1


def test_stats_summarises_the_community():
    rows = [_row("A", "game-dev", "commercial-only", 3, ["flux1-schnell", "background-removal"]),
            _row("B", "game-dev", "any", 1, ["flux1-schnell"])]
    summary = stats(rows)
    assert summary["workflows"] == 2 and summary["votes"] == 4
    assert summary["commercial_only"] == 1
    assert summary["by_industry"]["game-dev"] == 2
    assert summary["top_bricks"]["flux1-schnell"] == 2


def test_export_import_round_trip():
    spec = concept_spec()
    restored = import_spec(export_spec(spec))
    assert restored is not None
    assert restored.to_dict() == spec.to_dict()


def test_import_accepts_a_published_entry_or_a_bare_spec():
    spec = concept_spec()
    entry = {"slug": spec.slug, "spec": spec.to_dict(), "author": "someone"}
    assert import_spec(json.dumps(entry)).slug == spec.slug
    assert import_spec(json.dumps(spec.to_dict())).slug == spec.slug
    assert import_spec("not json") is None
    assert import_spec('{"hello": 1}') is None


def test_spec_from_entry_carries_the_author():
    spec = concept_spec()
    entry = {"spec": spec.to_dict(), "author": "jkorstad"}
    restored = spec_from_entry(entry)
    assert restored is not None and restored.author == "jkorstad"


def test_voting_without_a_token_degrades_clearly():
    result = hub.vote("concept-to-sprite", username="someone", token=None, direction=1)
    assert result.ok is False
    assert "attributable" in result.message


def test_dataset_url_points_at_the_real_thing():
    assert hub.dataset_url("my-flow").endswith("/workflows/my-flow.json")
    assert hub.DATASET_ID in hub.dataset_url()