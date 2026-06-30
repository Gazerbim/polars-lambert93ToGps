import math
import polars as pl

# Constants: Lambert93 / IAG GRS80
_E = 0.081819191042816
_LAMBDA_0 = 3.0 * math.pi / 180.0
_N = 0.725607765053267
_C = 11754255.426096
_XS = 700000.0
_YS = 12655612.049876
_NEWTON_ITERATIONS = 7


def lambert93_to_wgs84(
    x: str | pl.Expr,
    y: str | pl.Expr,
) -> tuple[pl.Expr, pl.Expr]:
    """
    Converts Lambert93 (EPSG:2154) coordinates into WGS84
    geographic coordinates (longitude/latitude in decimal degrees),
    using pure Polars (no numpy, no pyproj), via vectorized
    expressions that can be run directly inside a with_columns().

    Algorithm: Lambert conformal conic projection, based on the
    GRS80 ellipsoid, following the official IGN method
    (ALG0004 algorithm - "Transformation from Lambert coordinates
    to geographic coordinates").

    Steps:
        1. Recenter the (X, Y) coordinates relative to the
           projection pole (Xs, Ys).
        2. Convert to polar coordinates (radius R, angle gamma).
        3. Directly compute longitude from the polar angle and
           the central meridian.
        4. Compute the isometric latitude L from the radius R.
        5. Iteratively solve (Newton's method, 7 iterations) for
           the geographic latitude phi from L, accounting for the
           eccentricity of the GRS80 ellipsoid.
    Returns:
        A tuple (longitude_expr, latitude_expr):
            - longitude_expr: Polars expression for longitude,
              in decimal degrees, aliased "longitude".
            - latitude_expr: Polars expression for latitude,
              in decimal degrees, aliased "latitude".
        These expressions must be passed to df.with_columns([...])
        to be evaluated across the whole DataFrame.

    Notes:
        - Null input values (X or Y) automatically propagate as
          null output values (no manual error handling needed).
        - Expected precision: on the order of a micrometer after
          7 iterations, more than sufficient for application-level
          use (map display, geographic filtering).
        - For certified geodetic precision (official exports,
          cadastral use), prefer pyproj with official transformation
          grids (NTF/RGF93).
    Args:
        x: column name or Polars expression for the Lambert93 easting (X).
        y: column name or Polars expression for the Lambert93 northing (Y).
    """
    x_expr = pl.col(x) if isinstance(x, str) else x
    y_expr = pl.col(y) if isinstance(y, str) else y

    delta_x = x_expr - _XS
    delta_y = y_expr - _YS
    R = (delta_x.pow(2) + delta_y.pow(2)).sqrt()
    gamma = pl.arctan2(-delta_x, -delta_y)

    longitude = (_LAMBDA_0 + gamma / _N) * 180.0 / math.pi

    L = -1.0 / _N * (R / _C).log()
    phi = 2.0 * L.exp().arctan() - math.pi / 2.0
    for _ in range(_NEWTON_ITERATIONS):
        sin_phi = phi.sin()
        term = ((1.0 + _E * sin_phi) / (1.0 - _E * sin_phi)).pow(_E / 2.0)
        phi = 2.0 * (L.exp() * term).arctan() - math.pi / 2.0
    latitude = phi * 180.0 / math.pi

    return longitude.alias("longitude"), latitude.alias("latitude")
