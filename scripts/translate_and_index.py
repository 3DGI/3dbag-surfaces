"""
Translate CityJSONL features to a global coordinate transform and build a building adjacency index.

Usage:
    uv run --with geopandas scripts/translate_and_index.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import struct
import sqlite3
from pathlib import Path

import shapely.wkb
from shapely.ops import unary_union


ROOT = Path(__file__).parent.parent
DATA = ROOT / "tests" / "data"
OUTPUT = DATA / "3DBAG"
SCALE = 0.001  # fixed: both source files and metadata share this scale


def parse_tile_id(path: Path) -> tuple[int, int, int]:
    """Extract (z, x, y) from filename like '10-434-716.city.jsonl'."""
    m = re.match(r"(\d+)-(\d+)-(\d+)\.city\.jsonl", path.name)
    if not m:
        raise ValueError(f"Cannot parse tile id from {path.name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def compute_offsets(old_translate: list[float], new_translate: list[float]) -> list[int]:
    """Integer vertex offsets to convert from old local to new global transform."""
    return [round((old_translate[i] - new_translate[i]) / SCALE) for i in range(3)]


def retranslate_vertices(vertices: list[list[int]], offsets: list[int]) -> list[list[int]]:
    return [[v[0] + offsets[0], v[1] + offsets[1], v[2] + offsets[2]] for v in vertices]


def process_jsonl(path: Path, new_translate: list[float], output_root: Path) -> None:
    z, x, y = parse_tile_id(path)

    with path.open() as f:
        lines = f.readlines()

    header = json.loads(lines[0])
    old_translate: list[float] = header["transform"]["translate"]
    offsets = compute_offsets(old_translate, new_translate)

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        feature: dict = json.loads(line)
        if feature.get("type") != "CityJSONFeature":
            continue

        fid: str = feature["id"]
        feature["vertices"] = retranslate_vertices(feature["vertices"], offsets)

        out_dir = output_root / "crop_reconstruct" / str(z) / str(x) / str(y) / "objects" / fid / "reconstruct"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{fid}.city.jsonl"
        out_file.write_text(json.dumps(feature, separators=(",", ":")) + "\n")


def read_gpkg_geometries(gpkg_path: Path) -> list[tuple[str, object]]:
    """Return list of (identificatie, shapely_geometry) from pand layer."""
    conn = sqlite3.connect(gpkg_path)
    rows = conn.execute("SELECT identificatie, geom FROM pand WHERE geom IS NOT NULL").fetchall()
    conn.close()

    result = []
    for identificatie, geom_bytes in rows:
        if geom_bytes is None:
            continue
        # GeoPackage binary: 2-byte magic, 1-byte version, 1-byte flags, 4-byte srs_id,
        # then optional envelope, then WKB.
        flags = geom_bytes[3]
        envelope_indicator = (flags >> 1) & 0x07
        envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
        envelope_size = envelope_sizes.get(envelope_indicator, 0)
        wkb_offset = 8 + envelope_size
        geom = shapely.wkb.loads(bytes(geom_bytes[wkb_offset:]))
        result.append((identificatie, geom))
    return result


def build_adjacency(gpkg_paths: list[Path]) -> list[tuple[str, str]]:
    """Return sorted list of (id, adjacent_id) pairs from buffered intersection."""
    all_features: list[tuple[str, object]] = []
    for path in gpkg_paths:
        all_features.extend(read_gpkg_geometries(path))

    pairs: set[tuple[str, str]] = set()
    for i, (id_a, geom_a) in enumerate(all_features):
        buffered = geom_a.buffer(0.1)
        for j, (id_b, geom_b) in enumerate(all_features):
            if i == j or id_a == id_b:
                continue
            if buffered.intersects(geom_b):
                pair = (min(id_a, id_b), max(id_a, id_b))
                pairs.add(pair)

    # Expand into directed pairs sorted for reproducibility
    directed: list[tuple[str, str]] = []
    for a, b in sorted(pairs):
        directed.append((a, b))
        directed.append((b, a))
    directed.sort()
    return directed


def main() -> None:
    metadata_path = DATA / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    new_translate: list[float] = metadata["transform"]["translate"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(metadata_path, OUTPUT / "metadata.json")

    jsonl_files = sorted(DATA.glob("*.city.jsonl"))
    for jsonl_path in jsonl_files:
        print(f"Processing {jsonl_path.name}...")
        process_jsonl(jsonl_path, new_translate, OUTPUT)

    gpkg_files = sorted(DATA.glob("*.gpkg"))
    print(f"Building adjacency index from {len(gpkg_files)} GPKG files...")
    pairs = build_adjacency(gpkg_files)

    adjacency_path = DATA / "adjacency.csv"
    with adjacency_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identificatie", "adjacent_identificatie"])
        writer.writerows(pairs)

    print(f"Done. {len(pairs)} adjacency rows written to {adjacency_path}")
    feature_count = sum(1 for _ in OUTPUT.rglob("*.city.jsonl"))
    print(f"{feature_count} feature files written under {OUTPUT}")


if __name__ == "__main__":
    main()
