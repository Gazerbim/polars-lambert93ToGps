# polars-lambert93

Pure Polars expressions to convert Lambert93 (EPSG:2154) coordinates 
to WGS84 (longitude/latitude), with no numpy or pyproj dependency.

## Why?

`pyproj` is the standard tool for coordinate transformation, but it
requires converting Polars Series to numpy arrays and back, breaking
lazy evaluation and adding overhead. This library keeps everything
inside the Polars/Arrow engine for better performance on large
datasets, especially in lazy pipelines.

## Installation

\`\`\`bash
pip install polars-lambert93
\`\`\`

## Usage

\`\`\`python
import polars as pl
from polars_lambert93 import lambert93_to_wgs84

df = pl.DataFrame({"x": [700000.0], "y": [6600000.0]})
lon_expr, lat_expr = lambert93_to_wgs84("x", "y")
df = df.with_columns([lon_expr, lat_expr])
\`\`\`

## Accuracy

Validated against `pyproj` to sub-micrometer precision (7 Newton
iterations). For certified geodetic precision (official exports,
cadastral use), prefer `pyproj` with official NTF/RGF93 grids.

## Benchmarks

[Insérez vos chiffres sur 43M lignes ici, ça donnera de la crédibilité]
