import json
import math
import csv
import gzip
import os
import pathlib
from typing import Sequence

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas
import rtree.index
import scipy.spatial as ss
from pymeshfix import MeshFix
from shapely.geometry import Polygon, box
from shapely import MultiPolygon
from pgutils import PostgresConnection
from psycopg import sql

from party_walls import cityjson
from party_walls import geometry


def get_bearings(values, num_bins, weights):
    """Divides the values depending on the bins"""

    n = num_bins * 2

    bins = np.arange(n + 1) * 360 / n

    count, bin_edges = np.histogram(values, bins=bins, weights=weights)

    # move last bin to front, so eg 0.01° and 359.99° will be binned together
    count = np.roll(count, 1)
    bin_counts = count[::2] + count[1::2]

    # because we merged the bins, their edges are now only every other one
    bin_edges = bin_edges[range(0, len(bin_edges), 2)]

    return bin_counts, bin_edges


def get_wall_bearings(dataset, num_bins):
    """Returns the bearings of the azimuth angle of the normals for vertical
    surfaces of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_data:
        wall_idxs = [s == "WallSurface" for s in dataset.cell_data["semantics"]]
    else:
        wall_idxs = [n[2] == 0 for n in normals]

    normals = normals[wall_idxs]

    azimuth = [point_azimuth(n) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_data["Area"][wall_idxs]

    return get_bearings(azimuth, num_bins, surface_areas)


def get_roof_bearings(dataset, num_bins):
    """Returns the bearings of the (vertical surfaces) of a dataset"""

    normals = dataset.face_normals

    if "semantics" in dataset.cell_data:
        roof_idxs = [s == "RoofSurface" for s in dataset.cell_data["semantics"]]
    else:
        roof_idxs = [n[2] > 0 for n in normals]

    normals = normals[roof_idxs]

    xz_angle = [azimuth(n[0], n[2]) for n in normals]
    yz_angle = [azimuth(n[1], n[2]) for n in normals]

    sized = dataset.compute_cell_sizes()
    surface_areas = sized.cell_data["Area"][roof_idxs]

    xz_counts, bin_edges = get_bearings(xz_angle, num_bins, surface_areas)
    yz_counts, bin_edges = get_bearings(yz_angle, num_bins, surface_areas)

    return xz_counts, yz_counts, bin_edges


def orientation_plot(
        bin_counts,
        bin_edges,
        num_bins=36,
        title=None,
        title_y=1.05,
        title_font=None,
        show=False
):
    if title_font is None:
        title_font = {"family": "DejaVu Sans", "size": 12, "weight": "bold"}

    width = 2 * np.pi / num_bins

    positions = np.radians(bin_edges[:-1])

    radius = bin_counts / bin_counts.sum()

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_ylim(top=radius.max())

    # configure the y-ticks and remove their labels
    ax.set_yticks(np.linspace(0, radius.max(), 5))
    ax.set_yticklabels(labels="")

    # configure the x-ticks and their labels
    xticklabels = ["N", "", "E", "", "S", "", "W", ""]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=xticklabels)
    ax.tick_params(axis="x", which="major", pad=-2)

    # draw the bars
    ax.bar(
        positions,
        height=radius,
        width=width,
        align="center",
        bottom=0,
        zorder=2
    )

    if title:
        ax.set_title(title, y=title_y, fontdict=title_font)

    if show:
        plt.show()

    return plt


def get_surface_plot(
        dataset,
        num_bins=36,
        title=None,
        title_y=1.05,
        title_font=None
):
    """Returns a plot for the surface normals of a polyData"""

    bin_counts, bin_edges = get_wall_bearings(dataset, num_bins)

    return orientation_plot(bin_counts, bin_edges)


def azimuth(dx, dy):
    """Returns the azimuth angle for the given coordinates"""

    return (math.atan2(dx, dy) * 180 / np.pi) % 360


def point_azimuth(p):
    """Returns the azimuth angle of the given point"""

    return azimuth(p[0], p[1])


def point_zenith(p):
    """Return the zenith angle of the given 3d point"""

    z = [0.0, 0.0, 1.0]

    cosine_angle = np.dot(p, z) / (np.linalg.norm(p) * np.linalg.norm(z))
    angle = np.arccos(cosine_angle)

    return (angle * 180 / np.pi) % 360


def compute_stats(values, percentile=90, percentage=75):
    """
    Returns the stats (mean, median, max, min, range etc.) for a set of values.
    """
    hDic = {'Mean': np.mean(values), 'Median': np.median(values),
            'Max': max(values), 'Min': min(values),
            'Range': (max(values) - min(values)),
            'Std': np.std(values)}
    m = max([values.count(a) for a in values])
    if percentile:
        hDic['Percentile'] = np.percentile(values, percentile)
    if percentage:
        hDic['Percentage'] = (percentage / 100.0) * hDic['Range'] + hDic['Min']
    if m > 1:
        hDic['ModeStatus'] = 'Y'
        modeCount = [x for x in values if values.count(x) == m][0]
        hDic['Mode'] = modeCount
    else:
        hDic['ModeStatus'] = 'N'
        hDic['Mode'] = np.mean(values)
    return hDic


def add_value(dict, key, value):
    """Does dict[key] = dict[key] + value"""

    if key in dict:
        dict[key] = dict[key] + value
    else:
        dict[key] = value


def convexhull_volume(points):
    """Returns the volume of the convex hull"""

    try:
        return ss.ConvexHull(points).volume
    except:
        return 0


def boundingbox_volume(points):
    """Returns the volume of the bounding box"""

    minx = min(p[0] for p in points)
    maxx = max(p[0] for p in points)
    miny = min(p[1] for p in points)
    maxy = max(p[1] for p in points)
    minz = min(p[2] for p in points)
    maxz = max(p[2] for p in points)

    return (maxx - minx) * (maxy - miny) * (maxz - minz)


def get_errors_from_report(report, objid, cm):
    """Return the report for the feature of the given obj"""

    if not "features" in report:
        return []

    fid = objid

    obj = cm["CityObjects"][objid]
    primidx = 0

    if not "geometry" in obj or len(obj["geometry"]) == 0:
        return []

    if "parents" in obj:
        parid = obj["parents"][0]

        primidx = cm["CityObjects"][parid]["children"].index(objid)
        fid = parid

    for f in report["features"]:
        if f["id"] == fid:
            if "errors" in f["primitives"][primidx]:
                return list(
                    map(lambda e: e["code"], f["primitives"][primidx]["errors"]))
            else:
                return []

    return []


def validate_report(report, cm):
    """Returns true if the report is actually for this file"""

    # TODO: Actually validate the report and that it corresponds to this cm
    return True


def tree_generator_function(building_meshes):
    for i, (bid, mesh) in enumerate(building_meshes.items()):
        xmin, ymin, zmin = np.min(mesh.points, axis=0)
        xmax, ymax, zmax = np.max(mesh.points, axis=0)
        yield (i, (xmin, ymin, zmin, xmax, ymax, zmax), bid)


def get_neighbours(building_meshes, bid, r):
    """Return the neighbours of the given building"""

    xmin, ymin, zmin = np.min(building_meshes[bid].points, axis=0)
    xmax, ymax, zmax = np.max(building_meshes[bid].points, axis=0)
    bids = [n.object
            for n in r.intersection((xmin,
                                     ymin,
                                     zmin,
                                     xmax,
                                     ymax,
                                     zmax),
                                    objects=True)
            if n.object != bid]

    # if len(bids) == 0:
    #     bids = [n.object for n in r.nearest((xmin, ymin, zmin, xmax, ymax, zmax), 5, objects=True) if n.object != bid]

    return bids


class StatValuesBuilder:

    def __init__(self, values, indices_list) -> None:
        self.__values = values
        self.__indices_list = indices_list

    def compute_index(self, index_name):
        """Returns True if the given index is supposed to be computed"""

        return self.__indices_list is None or index_name in self.__indices_list

    def add_index(self, index_name, index_func):
        """Adds the given index value to the dict"""

        if self.compute_index(index_name):
            self.__values[index_name] = index_func()
        else:
            self.__values[index_name] = "NC"


def filter_lod(cm, lod='2.2'):
    for co_id in cm["CityObjects"]:
        co = cm["CityObjects"][co_id]

        new_geom = []

        for geom in co["geometry"]:
            if str(geom["lod"]) == str(lod):
                new_geom.append(geom)

        co["geometry"] = new_geom


def process_building(building,
                     building_id,
                     filter,
                     repair,
                     plot_buildings,
                     vertices,
                     building_meshes,
                     neighbour_ids=[],
                     custom_indices=None,
                     goffset=None):
    if not filter is None and filter != building_id:
        return building_id, None

    # TODO: Add options for all skip conditions below

    # Skip if type is not Building or Building part
    if not building["type"] in ["Building", "BuildingPart"]:
        return building_id, None

    # Skip if no geometry
    if not "geometry" in building or len(building["geometry"]) == 0:
        return building_id, None

    geom = building["geometry"][0]

    mesh = cityjson.to_polydata(geom, vertices).clean()

    try:
        tri_mesh = building_meshes[building_id]
    except:
        print(f"{building_id} geometry parsing crashed! Omitting...")
        return building_id, {"type": building["type"]}

    if plot_buildings:
        print(f"Plotting {building_id}")
        tri_mesh.plot(show_grid=True)

    t_origin = np.min(tri_mesh.points, axis=0)

    if repair:
        mfix = MeshFix(tri_mesh)
        mfix.repair()

        fixed = mfix.mesh
    else:
        fixed = tri_mesh

    area, point_count, surface_count = geometry.area_by_surface(mesh)

    if "semantics" in geom:
        roof_points = geometry.get_points_of_type(mesh, "RoofSurface")
        ground_points = geometry.get_points_of_type(mesh, "GroundSurface")
    else:
        roof_points = []
        ground_points = []

    if len(roof_points) == 0:
        height_stats = compute_stats([0])
        ground_z = 0
    else:
        height_stats = compute_stats([v[2] for v in roof_points])
        if len(ground_points) > 0:
            ground_z = min([v[2] for v in ground_points])
        else:
            ground_z = mesh.bounds[4]

    if len(ground_points) > 0:
        shape = cityjson.to_shapely(geom, vertices)
    else:
        shape = cityjson.to_shapely(geom, vertices, ground_only=False)

    values = {
        "area_ground": area["GroundSurface"],
        "area_roof_flat": area["RoofSurfaceFlat"],
        "area_roof_sloped": area["RoofSurfaceSloped"],
    }

    if custom_indices is None or len(custom_indices) > 0:

        shared_area = 0
        shared_polys = []

        if len(neighbour_ids) > 0:
            # Get neighbour meshes
            n_meshes = [building_meshes[nid]
                        for nid in neighbour_ids]

            # Compute shared walls

            # need to translate to origin to make the clustering work well (both quality of results and performance)
            fixed.points -= t_origin
            for neighbour in n_meshes:
                neighbour.points -= t_origin

            walls = np.hstack([geometry.intersect_surfaces([fixed, neighbour])
                               for neighbour in n_meshes])

            shared_area = sum([wall["area"][0] for wall in walls])
            shared_polys = [Polygon(wall["pts"] + (t_origin + goffset)) for wall in
                            walls]
            # undo translate to not mess up future calculations with these geometries
            fixed.points += t_origin
            for neighbour in n_meshes:
                neighbour.points += t_origin

        builder = StatValuesBuilder(values, custom_indices)

        builder.add_index("area_shared_wall", lambda: shared_area)
        builder.add_index("area_exterior_wall",
                          lambda: area["WallSurface"] - shared_area)
        builder.add_index("shared_wall_geometry",
                          lambda: MultiPolygon([poly for poly in shared_polys]).wkt)

    return building_id, values


class CityModel:
    def __init__(self, cm) -> None:
        filter_lod(cm)
        self.cm = cm

        if "transform" in cm:
            s = cm["transform"]["scale"]
            t = cm["transform"]["translate"]
            self.verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                          for v in cm["vertices"]]
        else:
            self.verts = cm["vertices"]
        self.vertices = np.array(self.verts)


def city_stats(inputs: Sequence[str | os.PathLike[str]],
               dsn: str,
               break_on_error: bool = False,
               precision: int = 2,
               repair: bool = False,
               without_indices: bool = False,
               plot_buildings: bool = False,
               filter: str = None) -> pd.DataFrame:
    """Compute statistics for a set of CityJSON files, including party walls.

    Args:
        inputs: A sequence of CityJSON file paths
        dsn: PostgreSQL connection string
        break_on_error: Throw the Exceptions. If false, Exceptions are only logged.
        precision: Round the returned DataFrame to the number of decimal places.
        repair:
        without_indices:
        plot_buildings:
        filter:

    Returns:
        A DataFrame of the statistics
    """
    cms = []
    # Check if we can connect to Postgres before we would start processing anything
    conn = PostgresConnection(dsn=dsn)

    for input in inputs:
        with open(input, "r") as fo:
            cms.append(CityModel(json.load(fo)))

    # we assume the first tile is the current tile we need to compute shared walls for
    active_tile_name = pathlib.Path(inputs[0]).name.replace(".city.json",
                                                            "").replace("-", "/")

    ge = cms[0].cm['metadata']['geographicalExtent']
    tile_bb = box(ge[0], ge[1], ge[3], ge[4])
    t_origin = [(p[0], p[1], 0) for p in tile_bb.centroid.coords]

    building_meshes = {}

    # convert geometries to polydata and select from the neighbour tiles only the ones that intersect with current tile boundary
    for i, cm in enumerate(cms):
        for coid, co in cm.cm['CityObjects'].items():
            if co['type'] == "BuildingPart":
                if i > 0:
                    minx, maxx, miny, maxy, _, _ = cityjson.get_bbox(co['geometry'][0],
                                                                     cm.verts)
                    if not tile_bb.intersects(box(minx, miny, maxx, maxy)):
                        continue
                mesh = cityjson.to_triangulated_polydata(co['geometry'][0],
                                                         cm.vertices).clean()
                mesh.points -= t_origin
                building_meshes[coid] = mesh

    if len(building_meshes) == 0:
        print("Aborting, no building meshes found...")
        return

    # Build the index of the city model
    p = rtree.index.Property()
    p.dimension = 3
    r = rtree.index.Index(tree_generator_function(building_meshes), properties=p)

    stats = {}

    for obj in cms[0].cm["CityObjects"]:
        if cms[0].cm["CityObjects"][obj]["type"] == "BuildingPart":
            neighbour_ids = get_neighbours(building_meshes, obj, r)

            indices_list = [] if without_indices else None

            try:
                obj, vals = process_building(cms[0].cm["CityObjects"][obj],
                                             obj,
                                             filter,
                                             repair,
                                             plot_buildings,
                                             cms[0].vertices,
                                             building_meshes,
                                             neighbour_ids,
                                             indices_list,
                                             goffset=t_origin)
                if not vals is None:
                    parent = cms[0].cm["CityObjects"][obj]["parents"][0]
                    for key, val in cms[0].cm["CityObjects"][parent][
                        "attributes"].items():
                        if key in ["identificatie", "status", "b3_pw_datum",
                                   "oorspronkelijkbouwjaar", "b3_volume_lod22"]:
                            if key == "b3_volume_lod22":
                                vals["volume"] = val
                            if key == "b3_pw_datum":
                                vals["pw_datum"] = val
                            else:
                                vals[key] = val
                    stats[obj] = vals
            except Exception as e:
                print(f"Problem with {obj}")
                if break_on_error:
                    raise e

    cm_ids = sql.Literal(list(cms[0].cm["CityObjects"].keys()))
    query = sql.SQL(
        """
        SELECT p.identificatie::text    AS identificatie
             , st_area(p.geometrie)     AS area_bag_source
        FROM lvbag.pandactueelbestaand p
        WHERE p.identificatie = ANY({cm_ids});
        """
    ).format(cm_ids=cm_ids)

    print("Building data frame...")
    df = pd.DataFrame.from_dict(stats, orient="index").round(precision)
    df.index.name = "id"
    df["identificatie"] = df["identificatie"].astype(str)

    print("Getting BAG footprint areas...")
    df = df.join(other=pd.DataFrame
                 .from_records(conn.get_dict(query))
                 .set_index("identificatie")
                 .round(precision),
                 on="identificatie", how="left")
    df['tile'] = active_tile_name
    df['area_ground_lost'] = df["area_bag_source"] - df["area_ground"]
    df['area_opening'] = df["area_exterior_wall"] * 0.2
    df['ratio_ground_to_volume'] = df["area_ground"] / df["volume"]
    df['ratio_roof_to_volume'] = (df["area_roof_flat"] + df["area_roof_sloped"]) / df[
        "volume"]
    df['ratio_exterior_wall_to_volume'] = df["area_exterior_wall"] / df["volume"]
    df['ratio_opening_to_volume'] = df["area_opening"] / df["volume"]
    return df


def process_files(inputs: Sequence[str],
                  output: str,
                  dsn: str,
                  gpkg: str = None,
                  break_on_error: bool = False,
                  jobs: int = 1,
                  precision: int = 2,
                  repair: bool = False,
                  without_indices: bool = False,
                  single_threaded: bool = False,
                  plot_buildings: bool = False,
                  filter: str = None):
    df = city_stats(inputs,
                    dsn,
                    break_on_error,
                    precision,
                    repair,
                    without_indices,
                    plot_buildings,
                    filter)

    if output is None:
        print(df)
    else:
        print(f"Writing shared walls output to {output}...")
        df.to_csv(output, sep=",", quoting=csv.QUOTE_ALL)

    if not gpkg is None:
        gdf = geopandas.GeoDataFrame(df, geometry="geometry")
        gdf.to_file(gpkg, driver="GPKG")

    print("Done")


# Assume semantic surfaces
@click.command()
@click.argument("inputs", nargs=-1, type=str)
@click.option('-o', '--output', type=click.Path(resolve_path=True,
                                                path_type=pathlib.Path))
@click.option('-g', '--gpkg')
@click.option('-f', '--filter')
@click.option('-r', '--repair', flag_value=True)
@click.option('-p', '--plot-buildings', flag_value=True)
@click.option('--without-indices', flag_value=True)
@click.option('-s', '--single-threaded', flag_value=True)
@click.option('-b', '--break-on-error', flag_value=True)
@click.option('-j', '--jobs', default=1)
@click.option('-dsn')
@click.option('--precision', default=2)
def main_cmd(inputs,
             output,
             gpkg,
             filter,
             repair,
             plot_buildings,
             without_indices,
             single_threaded,
             break_on_error,
             jobs,
             dsn,
             precision):
    process_files(
        inputs=inputs,
        output=output,
        gpkg=gpkg,
        filter=filter,
        repair=repair,
        plot_buildings=plot_buildings,
        without_indices=without_indices,
        single_threaded=single_threaded,
        break_on_error=break_on_error,
        jobs=jobs,
        dsn=dsn,
        precision=precision)


if __name__ == "__main__":
    main_cmd()
