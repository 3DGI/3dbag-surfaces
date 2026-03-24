"""Profile shared_walls() against real 3DBAG CityJSONFeature data.

Loads reconstructed buildings from a tile directory (default: the 415-building tile
10/564/624 from the 3dbag-pipeline integration test data), computes adjacency from LoD 0
footprints, and profiles each shared_walls() call with cProfile.

Outputs:
  - A cumulative stats table for each profiled building (stdout)
  - Per-building .prof files in --output-dir (openable with snakeviz)
  - An aggregate .prof over all buildings

Usage examples:

  # Profile all buildings in the default tile, save .prof files
  uv run scripts/profile_shared_walls.py

  # Profile specific slow buildings
  uv run scripts/profile_shared_walls.py \\
      --buildings NL.IMBAG.Pand.0307100000372977 NL.IMBAG.Pand.0307100000333121

  # Point at a different data directory
  uv run scripts/profile_shared_walls.py --data-dir /path/to/tile/objects

  # View results interactively
  uvx snakeviz profiles/aggregate.prof
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import io
import json
import pstats
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

# ---------------------------------------------------------------------------
# Default data location: 3dbag-pipeline integration test tile 10/564/624
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "3dbag-pipeline"
    / "tests"
    / "test_data"
    / "integration_party_walls"
    / "file_store_fastssd"
    / "3DBAG"
    / "crop_reconstruct"
    / "10"
    / "564"
    / "624"
    / "objects"
)

_NL_TRANSFORM = {
    "scale": [0.001, 0.001, 0.001],
    "translate": [171800.0, 472700.0, 0.0],
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_feature(building_dir: Path, transform: dict) -> dict | None:
    """Load a CityJSONFeature as the CityJSON dict that shared_walls() expects."""
    pand_id = building_dir.name
    jsonl_path = building_dir / "reconstruct" / f"{pand_id}.city.jsonl"
    if not jsonl_path.exists():
        return None
    feature = json.loads(jsonl_path.read_text())
    return {
        "type": "CityJSON",
        "version": "1.1",
        "transform": transform,
        "CityObjects": feature.get("CityObjects", {}),
        "vertices": feature.get("vertices", []),
        "_feature": feature,
    }


def find_building_part_id(cm: dict) -> str | None:
    for obj_id, obj in cm["CityObjects"].items():
        if obj.get("type") == "BuildingPart":
            return obj_id
    return next(iter(cm["CityObjects"]), None)


def build_adjacency(objects_dir: Path, transform: dict) -> dict[str, list[str]]:
    """Compute adjacency from LoD 0 footprints via Shapely STRtree."""
    from shapely import Polygon, STRtree

    scale = transform["scale"]
    translate = transform["translate"]
    polygons: list[Polygon] = []
    ids: list[str] = []

    for building_dir in sorted(objects_dir.iterdir()):
        if not building_dir.is_dir():
            continue
        pand_id = building_dir.name
        jsonl_path = building_dir / "reconstruct" / f"{pand_id}.city.jsonl"
        if not jsonl_path.exists():
            continue
        feature = json.loads(jsonl_path.read_text())
        building_obj = feature["CityObjects"].get(pand_id)
        if building_obj is None:
            continue
        lod0_geom = next(
            (g for g in building_obj.get("geometry", []) if g.get("lod") == "0"),
            None,
        )
        if lod0_geom is None:
            continue
        vertices = feature["vertices"]
        boundary = lod0_geom["boundaries"][0][0]
        coords = [
            (
                vertices[i][0] * scale[0] + translate[0],
                vertices[i][1] * scale[1] + translate[1],
            )
            for i in boundary
        ]
        if len(coords) >= 3:
            polygons.append(Polygon(coords))
            ids.append(pand_id)

    tree = STRtree(polygons)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for i, poly in enumerate(polygons):
        buffered = poly.buffer(0.1)
        for j in tree.query(buffered):
            if i != j and buffered.intersects(polygons[j]):
                adjacency[ids[i]].append(ids[j])
    return dict(adjacency)


def load_all_features(
    objects_dir: Path, transform: dict
) -> dict[str, tuple[dict, str]]:
    """Return {pand_id: (cm_dict, part_id)} for all buildings in objects_dir."""
    result: dict[str, tuple[dict, str]] = {}
    for building_dir in sorted(objects_dir.iterdir()):
        if not building_dir.is_dir():
            continue
        pand_id = building_dir.name
        cm = load_feature(building_dir, transform)
        if cm is None:
            continue
        part_id = find_building_part_id(cm)
        if part_id is None:
            continue
        result[pand_id] = (cm, part_id)
    return result


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def _print_stats(pr: cProfile.Profile, label: str, top_n: int = 20) -> None:
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(top_n)
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(s.getvalue())


def profile_building(
    pand_id: str,
    features: dict[str, tuple[dict, str]],
    adjacency: dict[str, list[str]],
    output_dir: Path,
    print_stats: bool = True,
) -> float:
    """Profile shared_walls() for one building; return wall-clock seconds."""
    from building_surfaces.walls import shared_walls

    cm, part_id = features[pand_id]
    adj_ids = [a for a in adjacency.get(pand_id, []) if a in features]
    adjacent = [(copy.deepcopy(features[a][0]), features[a][1]) for a in adj_ids]

    pr = cProfile.Profile()
    t0 = perf_counter()
    pr.enable()
    try:
        shared_walls(target=(copy.deepcopy(cm), part_id), adjacent=adjacent)
    except ValueError as exc:
        pr.disable()
        print(f"\n  ✗ skipped {pand_id}: {exc}")
        return 0.0
    pr.disable()
    elapsed = perf_counter() - t0

    label = f"{pand_id}  ({len(adj_ids)} adjacent, {elapsed:.2f}s)"
    if print_stats:
        _print_stats(pr, label)

    prof_path = output_dir / f"{pand_id}.prof"
    pr.dump_stats(str(prof_path))
    print(f"  → saved {prof_path.name}")

    return elapsed


def profile_all(
    pand_ids: list[str],
    features: dict[str, tuple[dict, str]],
    adjacency: dict[str, list[str]],
    output_dir: Path,
) -> None:
    """Profile all buildings into a single aggregate .prof."""
    from building_surfaces.walls import shared_walls

    pr = cProfile.Profile()
    timings: list[tuple[str, float]] = []
    skipped: list[str] = []
    total_t0 = perf_counter()

    for pand_id in pand_ids:
        cm, part_id = features[pand_id]
        adj_ids = [a for a in adjacency.get(pand_id, []) if a in features]
        adjacent = [(copy.deepcopy(features[a][0]), features[a][1]) for a in adj_ids]
        t0 = perf_counter()
        pr.enable()
        try:
            shared_walls(target=(copy.deepcopy(cm), part_id), adjacent=adjacent)
        except ValueError as exc:
            pr.disable()
            skipped.append(pand_id)
            sys.stdout.write(f"\r  skipped {pand_id}: {exc}\n")
            sys.stdout.flush()
            continue
        pr.disable()
        timings.append((pand_id, perf_counter() - t0))
        sys.stdout.write(f"\r  processed {len(timings)}/{len(pand_ids)}")
        sys.stdout.flush()

    total_elapsed = perf_counter() - total_t0
    print(f"\n  total wall-clock: {total_elapsed:.1f}s")
    if skipped:
        print(f"  {len(skipped)} buildings skipped (no LoD 2.2 geometry)")

    agg_path = output_dir / "aggregate.prof"
    pr.dump_stats(str(agg_path))
    _print_stats(pr, f"Aggregate — {len(pand_ids)} buildings", top_n=30)
    print(f"  → saved {agg_path}")

    # Print slowest buildings
    timings.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 slowest buildings:")
    print(f"  {'pand_id':<55} {'adj':>5} {'time':>8}")
    print(f"  {'-'*55} {'-'*5} {'-'*8}")
    for pid, t in timings[:10]:
        n_adj = len([a for a in adjacency.get(pid, []) if a in features])
        print(f"  {pid:<55} {n_adj:>5} {t:>8.2f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Path to tile objects/ directory (default: 3dbag-pipeline test data)",
    )
    parser.add_argument(
        "--buildings",
        nargs="+",
        metavar="PAND_ID",
        help="Profile specific building IDs (default: all buildings)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "profiles",
        help="Directory for .prof output files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of functions to show in pstats output",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: data dir not found: {args.data_dir}", file=sys.stderr)
        print("Pass --data-dir to point at a tile objects/ directory.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.data_dir} ...")
    t0 = perf_counter()
    features = load_all_features(args.data_dir, _NL_TRANSFORM)
    print(f"  loaded {len(features)} buildings in {perf_counter() - t0:.1f}s")

    print("Computing adjacency from LoD 0 footprints ...")
    t0 = perf_counter()
    adjacency = build_adjacency(args.data_dir, _NL_TRANSFORM)
    total_pairs = sum(len(v) for v in adjacency.values())
    print(f"  {total_pairs} directed pairs in {perf_counter() - t0:.1f}s")

    if args.buildings:
        unknown = [b for b in args.buildings if b not in features]
        if unknown:
            print(f"Error: unknown building IDs: {unknown}", file=sys.stderr)
            sys.exit(1)
        for pand_id in args.buildings:
            profile_building(pand_id, features, adjacency, args.output_dir,
                             print_stats=True)
    else:
        pand_ids = sorted(features.keys())
        print(f"\nProfiling all {len(pand_ids)} buildings (aggregate) ...")
        profile_all(pand_ids, features, adjacency, args.output_dir)


if __name__ == "__main__":
    main()
