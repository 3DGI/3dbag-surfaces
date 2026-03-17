"""Integration tests against real 3DBAG CityJSON data.

Reads adjacency pairs from adjacency.csv, computes shared_walls() for each
available building, and compares results to the pre-computed b3_* attributes
stored in the JSON files.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from shapely import MultiPolygon

from party_walls.walls import SharedWallResult, shared_walls, write_cityjsonfeature

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "3DBAG"
OBJECTS_DIR = DATA_DIR / "crop_reconstruct" / "10" / "434" / "716" / "objects"

_TRANSFORM: dict[str, Any] = json.loads((DATA_DIR / "metadata.json").read_text())[
    "transform"
]

_ADJACENCY: dict[str, list[str]] = defaultdict(list)
with (DATA_DIR / "adjacency.csv").open() as _f:
    for row in csv.DictReader(_f):
        _ADJACENCY[row["identificatie"]].append(row["adjacent_identificatie"])

_AVAILABLE: set[str] = {p.name for p in OBJECTS_DIR.iterdir() if p.is_dir()}
_TEST_IDS: list[str] = sorted(_AVAILABLE & _ADJACENCY.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cityjson(building_id: str, transform: dict[str, Any]) -> dict[str, Any]:
    jsonl_path = OBJECTS_DIR / building_id / "reconstruct" / f"{building_id}.city.jsonl"
    feature = json.loads(jsonl_path.read_text().splitlines()[0])
    return {
        "type": "CityJSON",
        "version": "2.0",
        "transform": transform,
        "CityObjects": feature["CityObjects"],
        "vertices": feature["vertices"],
    }


def _part_id(building_id: str, cm: dict[str, Any]) -> str:
    for obj_id, obj in cm["CityObjects"].items():
        if obj["type"] == "BuildingPart":
            return obj_id
    raise KeyError(f"No BuildingPart in {building_id}")


def _attrs(building_id: str, cm: dict[str, Any]) -> dict[str, Any]:
    return cm["CityObjects"][building_id]["attributes"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("building_id", _TEST_IDS)
def test_surface_areas(building_id: str) -> None:
    target_cm = _load_cityjson(building_id, _TRANSFORM)
    adj_ids = [a for a in _ADJACENCY[building_id] if a in _AVAILABLE]
    adjacents = [
        (_load_cityjson(a, _TRANSFORM), _part_id(a, _load_cityjson(a, _TRANSFORM)))
        for a in adj_ids
    ]

    result = shared_walls(
        target=(target_cm, _part_id(building_id, target_cm)),
        adjacent=adjacents,
    )
    attrs = _attrs(building_id, target_cm)

    assert result.area_exterior_wall == pytest.approx(
        attrs["b3_opp_buitenmuur"], rel=0.20, abs=1.0
    )
    assert result.area_ground == pytest.approx(attrs["b3_opp_grond"], rel=0.10, abs=1.0)
    assert result.area_roof_flat == pytest.approx(
        attrs["b3_opp_dak_plat"], rel=0.20, abs=1.0
    )
    assert result.area_roof_sloped == pytest.approx(
        attrs["b3_opp_dak_schuin"], rel=0.20, abs=1.0
    )
    assert result.area_shared_wall == pytest.approx(
        attrs["b3_opp_scheidingsmuur"], rel=0.20, abs=1.0
    )


@pytest.mark.parametrize("building_id", _TEST_IDS)
def test_shared_wall_detection(building_id: str) -> None:
    target_cm = _load_cityjson(building_id, _TRANSFORM)
    adj_ids = [a for a in _ADJACENCY[building_id] if a in _AVAILABLE]
    adjacents = [
        (_load_cityjson(a, _TRANSFORM), _part_id(a, _load_cityjson(a, _TRANSFORM)))
        for a in adj_ids
    ]
    missing_adj = len(_ADJACENCY[building_id]) - len(adj_ids)

    result = shared_walls(
        target=(target_cm, _part_id(building_id, target_cm)),
        adjacent=adjacents,
    )
    attrs = _attrs(building_id, target_cm)

    expected = attrs["b3_opp_scheidingsmuur"] > 0
    actual = result.area_shared_wall > 0

    assert expected == actual, (
        f"detection mismatch: computed={'yes' if actual else 'no'} "
        f"(area={result.area_shared_wall:.1f}), "
        f"3DBAG={'yes' if expected else 'no'} "
        f"(b3_opp_scheidingsmuur={attrs['b3_opp_scheidingsmuur']:.1f}), "
        f"missing_adjacents={missing_adj}"
    )


def test_write_output(tmp_path: Path) -> None:
    building_id = _TEST_IDS[0]
    jsonl_path = OBJECTS_DIR / building_id / "reconstruct" / f"{building_id}.city.jsonl"
    feature = json.loads(jsonl_path.read_text().splitlines()[0])

    sentinel = SharedWallResult(
        area_shared_wall=1.0,
        area_exterior_wall=2.0,
        shared_wall_geometry=MultiPolygon(),
        area_ground=3.0,
        area_roof_flat=4.0,
        area_roof_sloped=5.0,
    )

    out_path = tmp_path / f"{building_id}.city.jsonl"
    write_cityjsonfeature(feature, sentinel, out_path)

    written = json.loads(out_path.read_text())
    building_obj = next(
        obj for obj in written["CityObjects"].values() if obj["type"] == "Building"
    )
    attrs = building_obj["attributes"]
    assert attrs["b3_opp_scheidingsmuur"] == 1.0
    assert attrs["b3_opp_buitenmuur"] == 2.0
    assert attrs["b3_opp_grond"] == 3.0
    assert attrs["b3_opp_dak_plat"] == 4.0
    assert attrs["b3_opp_dak_schuin"] == 5.0
