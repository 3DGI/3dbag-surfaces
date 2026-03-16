download:
  cd tests/data
  wget https://data.3dbag.nl/v20250903/tiles/9/444/728/9-444-728.city.json.gz
  wget https://data.3dbag.nl/v20250903/tiles/10/434/716/10-434-716.city.json.gz
  wget https://data.3dbag.nl/v20250903/tiles/10/434/716/10-434-716.gpkg.gz
  wget https://data.3dbag.nl/v20250903/tiles/9/444/728/9-444-728.gpkg.gz
  gunzip 9-444-728.gpkg.gz
  gunzip 10-434-716.gpkg.gz
  gunzip 9-444-728.city.json.gz
  gunzip 10-434-716.city.json.gz
  cjseq cat --file 9-444-728.city.json > 9-444-728.city.jsonl
  cjseq cat --file 10-434-716.city.json > 10-434-716.city.jsonl
  echo '{"CityObjects": {}, "metadata": {"referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/7415", "title": "3DBAG", "pointOfContact": {"contactName": "3DBAG Team", "role": "owner", "emailAddress": "info@3dbag.nl", "website": "https://3dbag.nl"}, "version": "v2026.02.21", "fullMetadataUrl": "https://data.3dbag.nl/metadata/v20260221/metadata.json"}, "transform": {"scale": [0.001, 0.001, 0.001], "translate": [171800.0, 472700.0, 0.0]}, "type": "CityJSON", "version": "2.0", "vertices": []}' > tests/data/metadata.json

prepare-data:
  uv run --with geopandas scripts/translate_and_index.py
