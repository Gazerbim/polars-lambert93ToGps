import math
import polars as pl
import pytest
from pyproj import Transformer
from polars_lambert93 import lambert93_to_wgs84

REFERENCE_TRANSFORMER = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

@pytest.mark.parametrize("x,y", [
    (700000.0, 6600000.0),   # Paris approx
    (840000.0, 6520000.0),   # Lyon approx
    (500000.0, 6200000.0),   # Toulouse approx
])
def test_matches_pyproj(x, y):
    df = pl.DataFrame({"x": [x], "y": [y]})
    lon_expr, lat_expr = lambert93_to_wgs84("x", "y")
    result = df.with_columns([lon_expr, lat_expr])

    expected_lon, expected_lat = REFERENCE_TRANSFORMER.transform(x, y)

    assert result["longitude"][0] == pytest.approx(expected_lon, abs=1e-6)
    assert result["latitude"][0] == pytest.approx(expected_lat, abs=1e-6)


def test_null_propagation():
    df = pl.DataFrame({"x": [700000.0, None], "y": [6600000.0, None]})
    lon_expr, lat_expr = lambert93_to_wgs84("x", "y")
    result = df.with_columns([lon_expr, lat_expr])

    assert result["longitude"][1] is None
    assert result["latitude"][1] is None


def test_lazyframe_compatible():
    lazy_df = pl.LazyFrame({"x": [700000.0], "y": [6600000.0]})
    lon_expr, lat_expr = lambert93_to_wgs84("x", "y")
    result = lazy_df.with_columns([lon_expr, lat_expr]).collect()
    assert result["longitude"][0] is not None
