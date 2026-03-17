"""Module with functions for 3D geometrical operations"""

from typing import Any

import mapbox_earcut as earcut
import numpy as np
import numpy.typing as npt
import pyvista as pv
from shapely.geometry import MultiPolygon, Polygon


def surface_normal(poly: npt.ArrayLike) -> list[float]:
    n = [0.0, 0.0, 0.0]

    pts = np.asarray(poly)
    for i, v_curr in enumerate(pts):
        v_next = pts[(i + 1) % len(pts)]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    if all([c == 0 for c in n]):
        raise ValueError("No normal. Possible colinear points!")

    norm = float(np.linalg.norm(n))
    return [float(c / norm) for c in n]


def axes_of_normal(
    normal: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Returns an x-axis and y-axis on a plane of the given normal"""
    n = np.asarray(normal, dtype=np.float64)
    x_axis: npt.NDArray[np.float64]
    if n[2] > 0.001 or n[2] < -0.001:
        x_axis = np.array([1.0, 0.0, -n[0] / n[2]])
    elif n[1] > 0.001 or n[1] < -0.001:
        x_axis = np.array([1.0, -n[0] / n[1], 0.0])
    else:
        x_axis = np.array([-n[1] / n[0], 1.0, 0.0])

    x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis: npt.NDArray[np.float64] = np.cross(n, x_axis)

    return x_axis, y_axis


def project_2d(
    points: npt.ArrayLike,
    normal: npt.ArrayLike,
    origin: npt.ArrayLike | None = None,
) -> list[list[float]]:
    pts = np.asarray(points)
    if origin is None:
        origin = pts[0]

    x_axis, y_axis = axes_of_normal(normal)

    return [
        [float(np.dot(p - origin, x_axis)), float(np.dot(p - origin, y_axis))]
        for p in pts
    ]


def triangulate(mesh: pv.PolyData) -> pv.PolyData:
    """Triangulates a mesh in the proper way"""

    final_mesh = pv.PolyData()
    n_cells = mesh.n_cells
    for i in range(n_cells):
        cell = mesh.get_cell(i)
        if cell.type not in [5, 6, 7, 9, 10]:
            continue

        pts = cell.points
        p = project_2d(pts, mesh.face_normals[i])
        result = earcut.triangulate_float32(
            np.array(p, dtype=np.float32),
            np.array([len(p)], dtype=np.uint32),
        ).astype(np.int64)

        t_count = len(result.reshape(-1, 3))
        triangles = np.hstack([[3] + list(t) for t in result.reshape(-1, 3)])

        new_mesh = pv.PolyData(pts, triangles)
        for k in mesh.cell_data:
            new_mesh[k] = [mesh.cell_data[k][i] for _ in range(t_count)]

        final_mesh = final_mesh + new_mesh

    return final_mesh


def triangulate_polygon(
    face: list[list[int]],
    vertices: npt.NDArray[np.float64],
    offset: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Returns the points and triangles for a given CityJSON polygon"""

    points = vertices[np.hstack(face)]
    normal = surface_normal(points)
    holes = [0]
    for ring in face:
        holes.append(len(ring) + holes[-1])
    holes = holes[1:]

    points_2d = project_2d(points, normal)

    result = earcut.triangulate_float32(
        np.array(points_2d, dtype=np.float32),
        np.array(holes, dtype=np.uint32),
    ).astype(np.int64)

    result += offset

    t_count = len(result.reshape(-1, 3))
    if t_count == 0:
        return points, np.array([], dtype=np.int64)
    triangles: npt.NDArray[np.int64] = np.hstack(
        [[3] + list(t) for t in result.reshape(-1, 3)]
    )

    return points, triangles


def plane_params(
    normal: npt.ArrayLike,
    origin: npt.ArrayLike,
    rounding: int = 2,
) -> npt.NDArray[np.float64]:
    """Returns the params (a, b, c, d) of the plane equation for the given
    normal and origin point.
    """
    a, b, c = np.round(np.asarray(normal, dtype=np.float64), 3)
    x0, y0, z0 = np.asarray(origin, dtype=np.float64)

    d: float = float(-(a * x0 + b * y0 + c * z0))

    if rounding >= 0:
        d = round(d, rounding)

    return np.array([a, b, c, d], dtype=np.float64)


def project_mesh(
    mesh: pv.PolyData,
    normal: npt.ArrayLike,
    origin: npt.ArrayLike,
) -> MultiPolygon:
    """Project the faces of a mesh to the given plane"""
    p = []
    for i in range(mesh.n_cells):
        pts = mesh.get_cell(i).points

        pts_2d = project_2d(pts, normal, origin)
        p.append(Polygon(pts_2d))

    return MultiPolygon(p).buffer(0)  # type: ignore[return-value]


def to_3d(
    points: npt.ArrayLike,
    normal: npt.ArrayLike,
    origin: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Returns the 3d coordinates of a 2d points from a given plane"""

    xa, ya = axes_of_normal(normal)

    mat = np.array([xa, ya])
    pts = np.asarray(points)

    result: npt.NDArray[np.float64] = np.dot(pts, mat) + np.asarray(origin)
    return result


__all__: list[Any] = [
    "axes_of_normal",
    "plane_params",
    "project_2d",
    "project_mesh",
    "surface_normal",
    "to_3d",
    "triangulate",
    "triangulate_polygon",
]
