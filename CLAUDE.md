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

`party-walls` is a Python library that computes shared and exterior wall metrics between adjacent CityJSON buildings.

### Data Flow

```
CityJSON dict (raw JSON)
  -> CityModel (LoD filter + coordinate transform via scale/translate)
  -> cityjson.to_triangulated_polydata()  (PyVista PolyData with semantic cell data)
  -> geometry.intersect_surfaces()        (coplanar clustering -> 2D Shapely intersection -> 3D lift)
  -> SharedWallResult                     (area metrics + MultiPolygon geometry)
```

### Key Modules

- **`walls.py`** — Public API. `CityModel` loads and transforms CityJSON; `shared_walls()` is the main entry point.
- **`cityjson.py`** — CityJSON parsing: extracts boundary data, converts to Shapely or PyVista meshes.
- **`geometry.py`** — Higher-level mesh operations: `intersect_surfaces()` (core algorithm), `area_by_surface()`, `cluster_meshes()`.
- **`helpers/geometry.py`** — Low-level 3D math: plane projection, normal computation, triangulation via `mapbox_earcut`.

### Design Notes

- **Semantic surfaces**: CityJSON semantics (`GroundSurface`, `WallSurface`, `RoofSurface`) are propagated as PyVista cell data and used throughout for filtering/classification.
- **Numerical stability**: Meshes are translated near the origin before intersection to avoid floating-point issues with large real-world coordinates (e.g. Dutch RD New).
- **Two-stage intersection**: (1) Agglomerative clustering groups coplanar faces across buildings; (2) Shapely computes polygon intersections on projected 2D planes.
- **Immutability**: Input CityJSON dicts are deep-copied before mutation, preserving caller data.
- **Package manager**: `uv` with `hatchling` build backend, `src/` layout.
