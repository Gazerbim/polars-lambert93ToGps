import random
import time

import polars as pl
import pytest
from pyproj import Transformer

from polars_lambert93ToGps import lambert93_to_wgs84


REFERENCE_TRANSFORMER = Transformer.from_crs(
    "EPSG:2154",
    "EPSG:4326",
    always_xy=True,
)

TOLERANCE = 1e-6


@pytest.mark.parametrize(
    "x,y",
    [
        (652469.02, 6862035.26),  # Paris
        (840000.0, 6520000.0),    # Lyon
        (500000.0, 6200000.0),    # Toulouse
        (342000.0, 7050000.0),    # Lille
        (120000.0, 6830000.0),    # Brest
        (1047000.0, 6840000.0),   # Strasbourg
        (1040000.0, 6270000.0),   # Nice
        (1170000.0, 6120000.0),   # Corse
    ],
)
def test_known_points_against_pyproj(x, y):
    df = pl.DataFrame({"x": [x], "y": [y]})

    result = df.with_columns(*lambert93_to_wgs84("x", "y"))

    expected_lon, expected_lat = REFERENCE_TRANSFORMER.transform(x, y)

    assert abs(result["longitude"][0] - expected_lon) < TOLERANCE
    assert abs(result["latitude"][0] - expected_lat) < TOLERANCE


def test_null_values():
    df = pl.DataFrame({
        "x": [700000.0, None, None, 700000.0],
        "y": [6600000.0, None, 6600000.0, None],
    })

    result = df.with_columns(*lambert93_to_wgs84("x", "y"))

    assert result["longitude"][0] is not None
    assert result["latitude"][0] is not None

    assert result["longitude"][1] is None
    assert result["latitude"][1] is None

    assert result["longitude"][2] is None
    assert result["latitude"][2] is None

    assert result["longitude"][3] is None
    assert result["latitude"][3] is None


def test_column_names():
    df = pl.DataFrame({
        "x": [700000.0],
        "y": [6600000.0],
    })

    result = df.with_columns(*lambert93_to_wgs84("x", "y"))

    assert "longitude" in result.columns
    assert "latitude" in result.columns


def test_expr_inputs():
    df = pl.DataFrame({
        "x": [700000.0],
        "y": [6600000.0],
    })

    result = df.with_columns(
        *lambert93_to_wgs84(
            pl.col("x"),
            pl.col("y"),
        )
    )

    assert "longitude" in result.columns
    assert "latitude" in result.columns


def test_lazyframe():
    df = pl.DataFrame({
        "x": [700000.0],
        "y": [6600000.0],
    })

    result = (
        df.lazy()
        .with_columns(*lambert93_to_wgs84("x", "y"))
        .collect()
    )

    assert result.height == 1
    assert "longitude" in result.columns
    assert "latitude" in result.columns


def test_random_points_against_pyproj():
    random.seed(42)

    n = 1000

    xs = [random.uniform(0, 1300000) for _ in range(n)]
    ys = [random.uniform(6000000, 7200000) for _ in range(n)]

    df = pl.DataFrame({"x": xs, "y": ys})

    result = df.with_columns(*lambert93_to_wgs84("x", "y"))

    expected_lons, expected_lats = REFERENCE_TRANSFORMER.transform(xs, ys)

    lon_errors = [abs(result["longitude"][i] - expected_lons[i]) for i in range(n)]
    lat_errors = [abs(result["latitude"][i] - expected_lats[i]) for i in range(n)]

    max_lon_error = max(lon_errors)
    max_lat_error = max(lat_errors)

    print(f"\nMax longitude error : {max_lon_error:e}")
    print(f"Max latitude error  : {max_lat_error:e}")

    assert max_lon_error < TOLERANCE, f"Max lon error {max_lon_error:e} exceeds tolerance"
    assert max_lat_error < TOLERANCE, f"Max lat error {max_lat_error:e} exceeds tolerance"


@pytest.mark.parametrize(
    "x,y",
    [
        (0, 6000000),
        (0, 7200000),
        (1300000, 6000000),
        (1300000, 7200000),
    ],
)
def test_boundary_values(x, y):
    df = pl.DataFrame({"x": [x], "y": [y]})

    result = df.with_columns(*lambert93_to_wgs84("x", "y"))

    lon = result["longitude"][0]
    lat = result["latitude"][0]

    assert lon is not None
    assert lat is not None
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90


def test_out_of_france_values():
    """
    Verifies that Lambert93 coordinates outside the France bounding box
    return numeric values (not null, not NaN), even if geographically invalid.
    """
    df = pl.DataFrame({"x": [0.0], "y": [0.0]})

    result = df.with_columns(*lambert93_to_wgs84("x", "y"))

    lon = result["longitude"][0]
    lat = result["latitude"][0]

    assert lon is not None
    assert lat is not None
    assert lon == lon      
    assert lat == lat
