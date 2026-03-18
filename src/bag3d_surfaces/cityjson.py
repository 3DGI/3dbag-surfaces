"""Module with functions for manipulating CityJSON data"""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
from shapely.geometry import MultiPolygon, Polygon

from bag3d_surfaces.helpers.geometry import (
    triangulate_polygon,
)

# A CityJSON geometry object (MultiSurface, Solid, etc.)
CityJSONGeom = dict[str, Any]


def get_surface_boundaries(geom: CityJSONGeom) -> list[Any]:
    """Returns the boundaries for all surfaces"""

    if geom["type"] == "MultiSurface" or geom["type"] == "CompositeSurface":
        return geom["boundaries"]  # type: ignore[no-any-return]
    elif geom["type"] == "Solid":
        return geom["boundaries"][0]  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Geometry type {geom['type']!r} not supported")


def get_points(geom: CityJSONGeom, verts: npt.ArrayLike) -> list[Any]:
    """Return the points of the geometry"""

    boundaries = get_surface_boundaries(geom)
    v = np.asarray(verts)

    f = [idx for ring in boundaries for idx in ring[0]]
    return [v[i].tolist() for i in f]


def to_shapely(
    geom: CityJSONGeom,
    vertices: npt.NDArray[np.float64],
    ground_only: bool = True,
) -> MultiPolygon:
    """Returns a shapely geometry of the footprint from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    if ground_only and "semantics" in geom:
        semantics = geom["semantics"]
        if geom["type"] == "MultiSurface":
            values = semantics["values"]
        else:
            values = semantics["values"][0]

        ground_idxs = [
            semantics["surfaces"][i]["type"] == "GroundSurface" for i in values
        ]

        boundaries = np.array(boundaries, dtype=object)[ground_idxs]

    shape = MultiPolygon(
        [Polygon([vertices[v] for v in boundary[0]]) for boundary in boundaries]
    )

    return shape.buffer(0)  # type: ignore[return-value]


def to_polydata(
    geom: CityJSONGeom,
    vertices: npt.NDArray[np.float64],
) -> pv.PolyData:
    """Returns the polydata mesh from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    f = [[len(r[0])] + r[0] for r in [f for f in boundaries]]
    faces = np.hstack(f)

    mesh = pv.PolyData(vertices, faces)

    if "semantics" in geom:
        semantics = geom["semantics"]
        if geom["type"] == "MultiSurface":
            values = semantics["values"]
        else:
            values = semantics["values"][0]

        mesh.cell_data["semantics"] = [  # type: ignore[assignment]
            semantics["surfaces"][i]["type"] for i in values
        ]

    return mesh


def to_triangulated_polydata(
    geom: CityJSONGeom,
    vertices: npt.NDArray[np.float64],
    clean: bool = True,
) -> pv.PolyData:
    """Returns the triangulated polydata mesh from a CityJSON geometry"""

    boundaries = get_surface_boundaries(geom)

    semantic_types: list[str] = []
    if "semantics" in geom:
        semantics = geom["semantics"]
        if geom["type"] == "MultiSurface":
            values = semantics["values"]
        else:
            values = semantics["values"][0]

        semantic_types = [semantics["surfaces"][i]["type"] for i in values]

    points: list[Any] = []
    triangles: list[Any] = []
    face_semantics: list[str] = []
    triangle_count = 0
    for fid, face in enumerate(boundaries):
        try:
            new_points, new_triangles = triangulate_polygon(face, vertices, len(points))
        except Exception:
            continue

        points.extend(new_points)
        triangles.extend(new_triangles)
        t_count = int(len(new_triangles) / 4)

        triangle_count += t_count

        if semantic_types:
            face_semantics.extend([semantic_types[fid] for _ in np.arange(t_count)])

    mesh = pv.PolyData(points, triangles)

    if semantic_types:
        mesh["semantics"] = face_semantics  # type: ignore[assignment]

    if clean:
        mesh = cast(pv.PolyData, mesh.clean())

    return mesh


def get_bbox(
    geom: CityJSONGeom,
    verts: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    pts = np.array(get_points(geom, verts))

    return np.hstack(  # type: ignore[return-value]
        [[np.min(pts[:, i]), np.max(pts[:, i])] for i in range(np.shape(pts)[1])]
    )
