"""Building surface area metrics for CityJSON building models."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
from pymeshfix import MeshFix
from shapely import MultiPolygon
from shapely.geometry import Polygon

from building_surfaces import cityjson, geometry

CityJSONDict = dict[str, Any]


def filter_lod(cm: CityJSONDict, lod: str = "2.2") -> None:
    for co_id in cm["CityObjects"]:
        co = cm["CityObjects"][co_id]
        if "geometry" in co:
            co["geometry"] = [g for g in co["geometry"] if str(g["lod"]) == str(lod)]


class CityModel:
    cm: CityJSONDict
    verts: list[list[float]]
    vertices: npt.NDArray[np.float64]

    def __init__(self, cm: CityJSONDict, lod: str = "2.2") -> None:
        filter_lod(cm, lod)
        self.cm = cm

        if "transform" in cm:
            s = cm["transform"]["scale"]
            t = cm["transform"]["translate"]
            self.verts = [
                [v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                for v in cm["vertices"]
            ]
        else:
            self.verts = cm["vertices"]
        self.vertices = np.array(self.verts, dtype=np.float64)


@dataclass
class SharedWallResult:
    area_shared_wall: float
    area_exterior_wall: float
    shared_wall_geometry: MultiPolygon
    area_ground: float
    area_roof_flat: float
    area_roof_sloped: float


def shared_walls(
    target: tuple[CityJSONDict, str],
    adjacent: list[tuple[CityJSONDict, str]],
    repair: bool = False,
    lod: str = "2.2",
) -> SharedWallResult:
    """Compute shared wall metrics between a target building and its neighbours.

    Args:
        target: (CityJSON dict, object_id) for the building to analyse.
        adjacent: List of (CityJSON dict, object_id) for neighbouring buildings.
        repair: Run pymeshfix hole-repair on the target mesh before intersecting.
        lod: Level of detail to use (default "2.2").

    Returns:
        SharedWallResult with areas and shared wall geometry.
    """
    target_cm_dict, target_id = target

    # Deep-copy so callers' dicts are not mutated by filter_lod
    target_model = CityModel(copy.deepcopy(target_cm_dict), lod)
    target_obj = target_model.cm["CityObjects"][target_id]

    if not target_obj.get("geometry"):
        raise ValueError(f"Target object {target_id!r} has no geometry at LoD {lod!r}")

    target_geom = target_obj["geometry"][0]
    target_mesh = cast(
        pv.PolyData,
        cityjson.to_polydata(target_geom, target_model.vertices).clean(),
    )
    target_tri_mesh = cast(
        pv.PolyData,
        cityjson.to_triangulated_polydata(target_geom, target_model.vertices).clean(),
    )

    if repair:
        mfix = MeshFix(target_tri_mesh)
        mfix.repair()
        target_tri_mesh = mfix.mesh

    adj_tri_meshes = []
    for adj_cm_dict, adj_id in adjacent:
        adj_model = CityModel(copy.deepcopy(adj_cm_dict), lod)
        adj_obj = adj_model.cm["CityObjects"][adj_id]
        if not adj_obj.get("geometry"):
            continue
        adj_tri_mesh = cast(
            pv.PolyData,
            cityjson.to_triangulated_polydata(
                adj_obj["geometry"][0], adj_model.vertices
            ).clean(),
        )
        adj_tri_meshes.append(adj_tri_mesh)

    # Translate to near-origin for numerical stability during intersection
    t_origin: npt.NDArray[np.float64] = np.min(target_tri_mesh.points, axis=0)
    target_tri_mesh.points -= t_origin
    for adj_mesh in adj_tri_meshes:
        adj_mesh.points -= t_origin

    shared_area = 0.0
    shared_polys: list[Polygon] = []

    for adj_mesh in adj_tri_meshes:
        walls = geometry.intersect_surfaces([target_tri_mesh, adj_mesh])
        for wall in walls:
            shared_area += float(wall["area"][0])
            shared_polys.append(Polygon(wall["pts"] + t_origin))

    # Undo translation (meshes may be reused by caller)
    target_tri_mesh.points += t_origin
    for adj_mesh in adj_tri_meshes:
        adj_mesh.points += t_origin

    area, _, _ = geometry.area_by_surface(target_mesh)

    return SharedWallResult(
        area_shared_wall=shared_area,
        area_exterior_wall=area["WallSurface"] - shared_area,
        shared_wall_geometry=MultiPolygon(shared_polys),
        area_ground=area["GroundSurface"],
        area_roof_flat=area["RoofSurfaceFlat"],
        area_roof_sloped=area["RoofSurfaceSloped"],
    )


def write_cityjsonfeature(
    feature: dict[str, Any],
    result: SharedWallResult,
    output_path: Path,
) -> None:
    """Write a CityJSONFeature with computed b3_* attributes injected."""
    for obj in feature["CityObjects"].values():
        if obj["type"] == "Building":
            obj.setdefault("attributes", {}).update(
                {
                    "b3_opp_scheidingsmuur": result.area_shared_wall,
                    "b3_opp_buitenmuur": result.area_exterior_wall,
                    "b3_opp_grond": result.area_ground,
                    "b3_opp_dak_plat": result.area_roof_flat,
                    "b3_opp_dak_schuin": result.area_roof_sloped,
                }
            )
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feature))
