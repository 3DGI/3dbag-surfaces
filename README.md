# party-walls

Computes shared and exterior wall metrics between adjacent CityJSON buildings.

## Installation

```bash
pip install .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```python
from party_walls.walls import CityModel, shared_walls, write_cityjsonfeature

result = shared_walls(
    target=(cityjson_dict, "building-part-id"),
    adjacent=[(adj_dict, "adj-part-id"), ...],
)

print(result.area_shared_wall)    # shared wall area in m²
print(result.area_exterior_wall)  # exterior wall area in m²
print(result.area_ground)         # ground surface area in m²
print(result.area_roof_flat)      # flat roof area in m²
print(result.area_roof_sloped)    # sloped roof area in m²
```

The `target` and each `adjacent` entry are `(CityJSON dict, object_id)` tuples where
`object_id` identifies a `BuildingPart` within that dict. Input dicts must include a
`transform` key with `scale` and `translate` arrays (standard CityJSON).

To write results back into a CityJSONFeature file:

```python
write_cityjsonfeature(feature_dict, result, Path("output/building.city.jsonl"))
```

This injects the computed values as `b3_*` attributes on the `Building` object and
writes the feature as a single JSON line.

## Result fields

| Field                | 3DBAG attribute         |
|----------------------|-------------------------|
| `area_shared_wall`   | `b3_opp_scheidingsmuur` |
| `area_exterior_wall` | `b3_opp_buitenmuur`     |
| `area_ground`        | `b3_opp_grond`          |
| `area_roof_flat`     | `b3_opp_dak_plat`       |
| `area_roof_sloped`   | `b3_opp_dak_schuin`     |

## Development

```bash
uv run pytest          # run tests
uv run ruff check src/ tests/   # lint
uv run pyright         # type check
uv run ruff format src/ tests/  # format
```

## Acknowledgement

This repo is a trimmed-down version of [tudelft3d/3d-building-metrics](https://github.com/tudelft3d/3d-building-metrics) .

