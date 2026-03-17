"""Integration tests for bag3d_surfaces.walls.shared_walls.

Geometry setup
--------------
Two box buildings sharing a wall at x = 10:

    Building 1 (target)       Building 2 (adjacent)
    (0,0,0) – (10,5,8)        (10,0,0) – (20,5,8)

Shared wall: x=10 plane, 5 m wide × 8 m tall → 40 m²
"""

import pytest
from shapely import MultiPolygon

from bag3d_surfaces.walls import shared_walls

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _box_cityjson(x0, y0, x1, y1, z0, z1, obj_id):
    """Return a minimal CityJSON dict for a box building."""
    verts = [
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1],  # 7
    ]

    # Six faces with outward-pointing normals (Newell's method ordering)
    boundaries = [
        [[0, 3, 2, 1]],  # Ground  – normal −z
        [[4, 5, 6, 7]],  # Roof    – normal +z
        [[0, 1, 5, 4]],  # South   – normal −y
        [[1, 2, 6, 5]],  # East    – normal +x
        [[2, 3, 7, 6]],  # North   – normal +y
        [[3, 0, 4, 7]],  # West    – normal −x
    ]
    semantics = {
        "surfaces": [
            {"type": "GroundSurface"},
            {"type": "RoofSurface"},
            {"type": "WallSurface"},
            {"type": "WallSurface"},
            {"type": "WallSurface"},
            {"type": "WallSurface"},
        ],
        "values": [0, 1, 2, 2, 2, 2],
    }

    parent_id = f"{obj_id}-building"
    return {
        "type": "CityJSON",
        "version": "2.0",
        "CityObjects": {
            parent_id: {
                "type": "Building",
                "children": [obj_id],
            },
            obj_id: {
                "type": "BuildingPart",
                "parents": [parent_id],
                "geometry": [
                    {
                        "type": "MultiSurface",
                        "lod": "2.2",
                        "boundaries": boundaries,
                        "semantics": semantics,
                    }
                ],
            },
        },
        "vertices": verts,
    }


@pytest.fixture
def target():
    cm = _box_cityjson(x0=0, y0=0, x1=10, y1=5, z0=0, z1=8, obj_id="part-1")
    return cm, "part-1"


@pytest.fixture
def adjacent():
    cm = _box_cityjson(x0=10, y0=0, x1=20, y1=5, z0=0, z1=8, obj_id="part-2")
    return cm, "part-2"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shared_wall_area(target, adjacent):
    result = shared_walls(target=target, adjacent=[adjacent])

    # Shared wall: 5 m × 8 m = 40 m²
    assert result.area_shared_wall == pytest.approx(40.0, rel=0.01)


def test_exterior_wall_area(target, adjacent):
    result = shared_walls(target=target, adjacent=[adjacent])

    # Total wall surface: 2×(10×8) + 2×(5×8) = 240 m²
    # Exterior = 240 − 40 = 200 m²
    assert result.area_exterior_wall == pytest.approx(200.0, rel=0.01)


def test_ground_area(target, adjacent):
    result = shared_walls(target=target, adjacent=[adjacent])

    # 10 m × 5 m = 50 m²
    assert result.area_ground == pytest.approx(50.0, rel=0.01)


def test_roof_flat_area(target, adjacent):
    result = shared_walls(target=target, adjacent=[adjacent])

    # Flat horizontal roof: 10 m × 5 m = 50 m²
    assert result.area_roof_flat == pytest.approx(50.0, rel=0.01)
    assert result.area_roof_sloped == pytest.approx(0.0, abs=0.01)


def test_shared_wall_geometry_type(target, adjacent):
    result = shared_walls(target=target, adjacent=[adjacent])

    assert isinstance(result.shared_wall_geometry, MultiPolygon)
    assert not result.shared_wall_geometry.is_empty


def test_no_adjacents_gives_zero_shared_wall(target):
    result = shared_walls(target=target, adjacent=[])

    assert result.area_shared_wall == 0.0
    assert result.shared_wall_geometry.is_empty
    # All wall area is exterior when there are no neighbours
    assert result.area_exterior_wall == pytest.approx(240.0, rel=0.01)


def test_input_dicts_not_mutated(target, adjacent):
    target_cm, target_id = target
    adjacent_cm, adjacent_id = adjacent

    # Capture original geometry list lengths before the call
    target_geom_count = len(target_cm["CityObjects"][target_id]["geometry"])
    adjacent_geom_count = len(adjacent_cm["CityObjects"][adjacent_id]["geometry"])

    shared_walls(target=target, adjacent=[adjacent])

    assert len(target_cm["CityObjects"][target_id]["geometry"]) == target_geom_count
    assert (
        len(adjacent_cm["CityObjects"][adjacent_id]["geometry"]) == adjacent_geom_count
    )
