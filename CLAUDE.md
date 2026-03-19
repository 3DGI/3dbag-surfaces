# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_shared_walls.py::test_shared_wall_area

# Lint
uv run ruff check src/ tests/

# Type check
uv run pyright

# Format
uv run ruff format src/ tests/
```

## Architecture

`building-surfaces` (`building_surfaces`) is a Python library that computes shared and exterior wall metrics between adjacent CityJSON buildings.

### Data Flow

```
CityJSON dict (raw JSON)
  -> CityModel (LoD filter + coordinate transform via scale/translate)
  -> cityjson.to_triangulated_polydata()  (PyVista PolyData with semantic cell data)
  -> geometry.intersect_surfaces()        (coplanar clustering -> 2D Shapely intersection -> 3D lift)
  -> SharedWallResult                     (area metrics + MultiPolygon geometry)
```

### Key Modules

- **`walls.py`** — Public API. `CityModel` loads and transforms CityJSON; `shared_walls()` is the main entry point; `write_cityjsonfeature()` injects computed `b3_*` attributes back into a CityJSONFeature and writes it to disk.
- **`cityjson.py`** — CityJSON parsing: extracts boundary data, converts to Shapely or PyVista meshes.
- **`geometry.py`** — Higher-level mesh operations: `intersect_surfaces()` (core algorithm), `area_by_surface()`, `cluster_meshes()`.
- **`helpers/geometry.py`** — Low-level 3D math: plane projection, normal computation, triangulation via `mapbox_earcut`.

### Key Types

`SharedWallResult` fields map to Dutch 3DBAG attributes:
- `area_shared_wall` → `b3_opp_scheidingsmuur`
- `area_exterior_wall` → `b3_opp_buitenmuur`
- `area_ground` → `b3_opp_grond`
- `area_roof_flat` / `area_roof_sloped` → `b3_opp_dak_plat` / `b3_opp_dak_schuin`

`RoofSurface` is split into flat vs. sloped using a 3° angle threshold from vertical (`sloped_angle_threshold` in `area_by_surface()`).

### Design Notes

- **Semantic surfaces**: CityJSON semantics (`GroundSurface`, `WallSurface`, `RoofSurface`) are propagated as PyVista cell data and used throughout for filtering/classification.
- **Numerical stability**: Meshes are translated near the origin before intersection to avoid floating-point issues with large real-world coordinates (e.g. Dutch RD New).
- **Two-stage intersection**: (1) Agglomerative clustering groups coplanar faces across buildings; (2) Shapely computes polygon intersections on projected 2D planes.
- **Clustering**: `cluster_meshes()` defaults to `old_cluster_method=True` (`cluster_faces_simple`), which drops the z-component (valid for vertical walls) and uses a distance-matrix approach on (nx, ny, d) plane params. The alternative (`cluster_faces_alternative`) separates angle and distance clustering via two `AgglomerativeClustering` passes.
- **Immutability**: Input CityJSON dicts are deep-copied before mutation, preserving caller data.
- **Package manager**: `uv` with `hatchling` build backend, `src/` layout.

### Integration Test Data

`tests/test_3dbag_integration.py` reads real 3DBAG CityJSON files from `tests/data/3DBAG/`. Required layout:
```
tests/data/3DBAG/
  metadata.json          # contains top-level "transform" (scale/translate)
  adjacency.csv          # columns: identificatie, adjacent_identificatie
  crop_reconstruct/10/434/716/objects/<building_id>/reconstruct/<building_id>.city.jsonl
```
Tests are parameterized over buildings present in both `adjacency.csv` and the objects directory.
