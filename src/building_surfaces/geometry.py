"""Module to manipulate geometry of pyvista meshes.

Derived from https://github.com/tudelft3d/3d-building-metrics (MIT License,
Copyright 2021 3D geoinformation research group at TU Delft). Modernised with
type annotations and extended with flat/sloped roof classification and an
alternative clustering method.
"""

from collections import defaultdict
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
from scipy.spatial import distance_matrix
from shapely import Polygon, intersects
from shapely.geometry.base import BaseGeometry
from sklearn.cluster import AgglomerativeClustering

from building_surfaces.helpers.geometry import plane_params, project_mesh, to_3d


def get_points_of_type(
    mesh: pv.PolyData,
    surface_type: str,
) -> npt.NDArray[np.float64]:
    """Returns the points that belong to the given surface type"""

    if "semantics" not in mesh.cell_data:
        return np.empty((0, 3), dtype=np.float64)

    idxs = [s == surface_type for s in mesh.cell_data["semantics"]]

    points = np.array(
        [mesh.get_cell(i).points for i in range(mesh.number_of_cells)], dtype=object
    )

    if all([not i for i in idxs]):
        return np.empty((0, 3), dtype=np.float64)

    return np.vstack(points[idxs])  # type: ignore[return-value]


def move_to_origin(
    mesh: pv.PolyData,
) -> tuple[pv.PolyData, npt.NDArray[np.float64]]:
    """Moves the object to the origin"""
    pts = mesh.points
    t: npt.NDArray[np.float64] = np.min(pts, axis=0)
    mesh.points = mesh.points - t

    return mesh, t


def extrude(shape: Polygon, min: float, max: float) -> pv.PolyData:
    """Create a pyvista mesh from a polygon"""

    points = np.array([[p[0], p[1], min] for p in shape.exterior.coords])
    mesh = pv.PolyData(points).delaunay_2d()

    if min == max:
        return cast(pv.PolyData, mesh)

    # Transform to 0, 0, 0 to avoid precision issues
    pts = mesh.points
    t = np.mean(pts, axis=0)
    mesh.points = mesh.points - t

    extruded = cast(pv.PolyData, mesh.extrude([0.0, 0.0, max - min], capping=True))

    # Transform back to origina coords
    # extruded.points = extruded.points + t

    return cast(pv.PolyData, extruded.clean().triangulate())


def area_by_surface(
    mesh: pv.PolyData,
    sloped_angle_threshold: float = 3,
    tri_mesh: pv.PolyData | None = None,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """Compute the area per semantic surface"""

    sloped_threshold = np.cos(np.radians(sloped_angle_threshold))

    area: dict[str, float] = {
        "GroundSurface": 0.0,
        "WallSurface": 0.0,
        "RoofSurfaceFlat": 0.0,
        "RoofSurfaceSloped": 0.0,
    }

    point_count: dict[str, int] = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0,
    }

    surface_count: dict[str, int] = {
        "GroundSurface": 0,
        "WallSurface": 0,
        "RoofSurface": 0,
    }

    # Compute the triangulated surfaces to fix issues with areas
    effective_tri: pv.PolyData = (
        tri_mesh if tri_mesh is not None else cast(pv.PolyData, mesh.triangulate())
    )

    if "semantics" in mesh.cell_data:
        # Compute area per surface type
        sized = cast(pv.PolyData, effective_tri.compute_cell_sizes())
        surface_areas = sized.cell_data["Area"]

        points_per_cell = np.array(
            [mesh.get_cell(i).n_points for i in range(mesh.number_of_cells)]
        )

        for surface_type in ["GroundSurface", "WallSurface", "RoofSurface"]:
            triangle_idxs_mask = [
                s == surface_type for s in effective_tri.cell_data["semantics"]
            ]
            triangle_idxs = [
                i
                for i, s in enumerate(effective_tri.cell_data["semantics"])
                if s == surface_type
            ]

            if surface_type == "RoofSurface":
                all_normals = sized.cell_normals
                for idx in triangle_idxs:
                    if all_normals[idx].dot([0, 0, 1]) < sloped_threshold:
                        area["RoofSurfaceSloped"] += float(surface_areas[idx])
                    else:
                        area["RoofSurfaceFlat"] += float(surface_areas[idx])
            else:
                area[surface_type] = float(sum(surface_areas[triangle_idxs_mask]))

            face_idxs = [s == surface_type for s in mesh.cell_data["semantics"]]

            point_count[surface_type] = int(sum(points_per_cell[face_idxs]))
            surface_count[surface_type] = int(sum(face_idxs))

    return area, point_count, surface_count


def face_planes(mesh: pv.PolyData) -> list[npt.NDArray[np.float64]]:
    """Return the params of all planes in a given mesh.

    Vectorised: uses mesh.face_normals and mesh.cell_centers() to avoid a
    per-cell Python loop over mesh.get_cell(i).
    """
    if mesh.n_cells == 0:
        return []
    normals = np.round(mesh.face_normals, 3)
    origins = mesh.cell_centers().points
    d = np.round(-np.sum(normals * origins, axis=1), 2)
    result = np.column_stack([normals, d])
    return list(result)


def cluster_meshes(
    meshes: list[pv.PolyData],
    angle_degree_threshold: float = 5,
    dist_threshold: float = 0.5,
    old_cluster_method: bool = True,
) -> tuple[list[npt.NDArray[np.intp]], int]:
    """Clusters the faces of the given meshes"""

    n_meshes = len(meshes)

    # Compute the "absolute" plane params for every face of the two meshes
    planes: list[list[npt.NDArray[np.float64]]] = [face_planes(mesh) for mesh in meshes]
    # convert to cosine distance value
    # cos_distance = 1 - cos_similarity
    # angle_rad = arccos(cos_similarity)
    # angle_deg = angle_rad * (180/pi)
    # Find the common planes between the two faces
    while [] in planes:
        planes.remove([])
    all_planes = np.concatenate(planes)
    if old_cluster_method:
        all_labels, n_clusters = cluster_faces_simple(all_planes)
    else:
        cos_dist_thres = 1 - np.cos((np.pi / 180) * angle_degree_threshold)
        all_labels, n_clusters = cluster_faces_alternative(
            all_planes, cos_dist_thres, dist_threshold
        )

    labels: list[npt.NDArray[np.intp]] = np.array_split(
        all_labels, [meshes[m].n_cells for m in range(n_meshes - 1)]
    )

    return labels, n_clusters


def cluster_faces_bucketed(
    data: npt.NDArray[np.float64],
    threshold: float = 0.1,
) -> tuple[npt.NDArray[np.intp], int]:
    """Cluster co-planar faces using hash-bucketing instead of an O(N²) distance matrix.

    Produces equivalent groupings to cluster_faces_simple for typical building
    geometry in O(N) time and space. The approach relies on the fact that truly
    co-planar faces (same wall plane) have nearly identical plane params (within
    floating-point noise << threshold), while distinct planes differ by much more
    than threshold.

    Mirrors cluster_faces_simple: drops the z-normal column (valid for vertical
    walls) and flips normals to the positive-x hemisphere before bucketing.
    """
    # Drop the z-normal column — valid for vertical WallSurface faces
    ndata = np.delete(data, 2, 1).astype(np.float64, copy=True)

    # Flip normals so both directions of the same plane map to the same bucket
    neg_x = ndata[:, 0] < 0
    ndata[neg_x] *= -1

    # Quantise to bins of size `threshold` and group by integer key
    scale = 1.0 / threshold
    keys = np.round(ndata * scale).astype(np.int32)

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        buckets[(int(key[0]), int(key[1]), int(key[2]))].append(i)

    labels: npt.NDArray[np.intp] = np.empty(len(data), dtype=np.intp)
    for cluster_id, indices in enumerate(buckets.values()):
        for idx in indices:
            labels[idx] = cluster_id

    return labels, len(buckets)


def cluster_faces_simple(
    data: npt.NDArray[np.float64],
    threshold: float = 0.1,
) -> tuple[npt.NDArray[np.intp], int]:
    """Clusters the given planes using an O(N²) distance matrix.

    Kept for reference and testing. cluster_faces_bucketed is the default
    in cluster_meshes and intersect_walls.
    """
    # we can delete the third column because it is all 0's for vertical planes
    ndata = np.delete(data, 2, 1)

    # flip normals so that they can not be pointing in opposite direction for same plane
    neg_x = ndata[:, 0] < 0
    ndata[neg_x, :] = ndata[neg_x, :] * -1

    dist_mat = distance_matrix(ndata, ndata)
    # dm2 = distance_matrix(ndata, -ndata)
    # dist_mat = np.minimum(dm1, dm2)

    clustering = AgglomerativeClustering(
        n_clusters=None,  # type: ignore[arg-type]
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    ).fit(dist_mat)

    return clustering.labels_, clustering.n_clusters_  # type: ignore[return-value]


def cluster_faces_alternative(
    data: npt.NDArray[np.float64],
    angle_threshold: float = 0.005,
    dist_threshold: float = 0.2,
) -> tuple[npt.NDArray[np.intp], int]:
    """Clusters the given planes"""

    def groupby(
        a: npt.NDArray[np.float64],
        clusterids: npt.NDArray[np.intp],
    ) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.intp]]:
        # Get argsort indices, to be used to sort a and clusterids in the next steps
        sidx = clusterids.argsort(kind="mergesort")
        a_sorted = a[sidx]
        clusterids_sorted = clusterids[sidx]

        # Get the group limit indices (start, stop of groups)
        cut_idx = np.flatnonzero(
            np.r_[True, clusterids_sorted[1:] != clusterids_sorted[:-1], True]
        )

        # Split input array with those start, stop ones
        out = [a_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])]
        return out, sidx

    ndata = np.array(data)

    # original method
    # dm1 = distance_matrix(ndata, ndata)
    # dm2 = distance_matrix(ndata, -ndata)

    # dist_mat = np.minimum(dm1, dm2)
    # clustering = AgglomerativeClustering(n_clusters=None,
    #                                      distance_threshold=threshold,
    #                                      affinity='precomputed',
    #                                      linkage='average').fit(dist_mat)

    # new method - angle clustering
    # pl_abc = ndata
    angle_clustering = AgglomerativeClustering(
        n_clusters=None,  # type: ignore[arg-type]
        metric="cosine",
        distance_threshold=angle_threshold,
        linkage="average",
    ).fit(ndata[:, :3])
    # group angle clusters
    angle_clusters, remap = groupby(ndata[:, 3:], angle_clustering.labels_)

    # get dist clusters for each angle cluster
    labels_: npt.NDArray[np.intp] = np.empty(0, dtype=int)
    min_label = 0
    for angle_cluster in angle_clusters:
        if angle_cluster.size == 1:
            labels_ = np.hstack((labels_, min_label))
            min_label += 1
        else:
            dist_clustering = AgglomerativeClustering(
                n_clusters=None,  # type: ignore[arg-type]
                metric="euclidean",
                distance_threshold=dist_threshold,
                linkage="average",
            ).fit(angle_cluster)
            labels_ = np.hstack((labels_, dist_clustering.labels_ + min_label))
            min_label = labels_.max() + 1

    # re order back to input data order
    n_planes = ndata.shape[0]
    labels: npt.NDArray[np.intp] = np.empty(n_planes, dtype=int)
    labels[remap] = labels_

    n_clusters = int((np.bincount(labels) != 0).sum())
    return labels, n_clusters


# ---------------------------------------------------------------------------
# Helpers shared by intersect_surfaces and intersect_walls
# ---------------------------------------------------------------------------


def _get_wall_mesh(mesh: pv.PolyData) -> pv.PolyData:
    """Extract WallSurface cells from a triangulated mesh."""
    return cast(
        pv.PolyData,
        mesh.remove_cells(
            [s != "WallSurface" for s in mesh.cell_data["semantics"]],
            inplace=False,
        ),
    )


def _collect_ring_area(
    areas: list[pv.PolyData],
    area: float,
    geom: BaseGeometry,
    normal: npt.ArrayLike,
    origin: npt.ArrayLike,
    subtract: bool = False,
) -> None:
    pts = to_3d(geom.coords, normal, origin)  # type: ignore[attr-defined]
    common_mesh = cast(
        pv.PolyData,
        pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts)))),
    )
    common_mesh["area"] = [-area] if subtract else [area]
    common_mesh["pts"] = pts
    areas.append(common_mesh)


def _collect_polygon_area(
    areas: list[pv.PolyData],
    geom: BaseGeometry,
    normal: npt.ArrayLike,
    origin: npt.ArrayLike,
) -> None:
    if geom.boundary.geom_type == "MultiLineString":
        _collect_ring_area(
            areas,
            float(geom.area),
            geom.boundary.geoms[0],  # type: ignore[attr-defined]
            normal,
            origin,
        )
        for hole in list(geom.boundary.geoms)[1:]:  # type: ignore[attr-defined]
            _collect_ring_area(areas, 0.0, hole, normal, origin, subtract=True)
    elif geom.boundary.geom_type == "LineString":
        _collect_ring_area(areas, float(geom.area), geom.boundary, normal, origin)


def _polygon_intersections(
    meshes_to_cluster: list[pv.PolyData],
    labels: list[npt.NDArray[np.intp]],
    n_clusters: int,
) -> list[pv.PolyData]:
    """Core polygon intersection loop shared by intersect_surfaces and intersect_walls."""
    n_meshes = len(meshes_to_cluster)
    areas: list[pv.PolyData] = []

    for plane in range(n_clusters):
        # For every common plane, extract the faces that belong to it
        idxs = [
            [i for i, p in enumerate(labels[m]) if p == plane] for m in range(n_meshes)
        ]

        if any(len(idx) == 0 for idx in idxs):
            continue

        msurfaces = [
            cast(
                pv.PolyData,
                mesh.extract_cells(idxs[i]).extract_surface(
                    algorithm="dataset_surface"
                ),
            )
            for i, mesh in enumerate(meshes_to_cluster)
        ]

        # Set the normal and origin point for a plane to project the faces
        origin = msurfaces[0].clean().points[0]
        normal = msurfaces[0].face_normals[0]

        if np.linalg.norm(normal) == 0:
            continue

        # Create the two 2D polygons by projecting the faces
        polys = [project_mesh(msurface, normal, origin) for msurface in msurfaces]

        # Intersect the 2D polygons
        inter: BaseGeometry = Polygon()
        poly_0 = polys[0]
        for i in range(1, len(polys)):
            if intersects(poly_0, polys[i]):
                inter = inter.union(poly_0.intersection(polys[i]))

        if inter.area > 0.001:
            if (
                inter.geom_type == "MultiPolygon"
                or inter.geom_type == "GeometryCollection"
            ):
                for geom in inter.geoms:  # type: ignore[attr-defined]
                    if geom.geom_type != "Polygon":
                        continue
                    _collect_polygon_area(areas, geom, normal, origin)
            elif inter.geom_type == "Polygon":
                _collect_polygon_area(areas, inter, normal, origin)

    return areas


# ---------------------------------------------------------------------------
# Public intersection API
# ---------------------------------------------------------------------------


def prepare_wall_mesh(
    mesh: pv.PolyData,
) -> tuple[pv.PolyData, list[npt.NDArray[np.float64]]]:
    """Extract wall cells and precompute plane params for a triangulated mesh.

    Call this once on the target mesh before iterating over adjacent buildings,
    then pass the returned values to intersect_walls() for each adjacent.  This
    avoids redundant per-adjacent reprocessing of the target's wall geometry.

    Returns:
        (wall_mesh, planes) where wall_mesh contains only WallSurface cells and
        planes is the list of plane parameters for those cells.
    """
    wall_mesh = _get_wall_mesh(mesh)
    planes = face_planes(wall_mesh)
    return wall_mesh, planes


def intersect_surfaces(
    meshes: list[pv.PolyData],
    onlywalls: bool = True,
) -> list[pv.PolyData]:
    """Return the intersection between the surfaces of multiple meshes.

    Note: first mesh is the target; following meshes are neighbors.
    """
    meshes_to_cluster = [_get_wall_mesh(m) for m in meshes] if onlywalls else meshes
    labels, n_clusters = cluster_meshes(meshes_to_cluster)
    return _polygon_intersections(meshes_to_cluster, labels, n_clusters)


def intersect_walls(
    target_wall_mesh: pv.PolyData,
    target_planes: list[npt.NDArray[np.float64]],
    adj_mesh: pv.PolyData,
) -> list[pv.PolyData]:
    """Intersect a precomputed target wall mesh against one adjacent mesh.

    Like intersect_surfaces([target_mesh, adj_mesh], onlywalls=True) but accepts
    precomputed target wall data from prepare_wall_mesh() to skip redundant target
    processing when the same target is intersected against multiple adjacents.
    """
    if not target_planes:
        return []
    adj_wall_mesh = _get_wall_mesh(adj_mesh)
    adj_planes = face_planes(adj_wall_mesh)
    if not adj_planes:
        return []
    all_planes = np.concatenate([target_planes, adj_planes])
    all_labels, n_clusters = cluster_faces_simple(all_planes)
    labels: list[npt.NDArray[np.intp]] = np.array_split(
        all_labels, [target_wall_mesh.n_cells]
    )
    return _polygon_intersections([target_wall_mesh, adj_wall_mesh], labels, n_clusters)


# Keep cluster_faces_bucketed available for experimentation but it is not used by default.
# See comments in cluster_meshes for context.
