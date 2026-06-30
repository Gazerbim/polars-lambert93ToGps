"""
Benchmark: polars-lambert93 vs pyproj
Run with: python benchmark.py
"""

import random
import time

import polars as pl
from pyproj import Transformer

from polars_lambert93ToGps import lambert93_to_wgs84


REFERENCE_TRANSFORMER = Transformer.from_crs(
    "EPSG:2154",
    "EPSG:4326",
    always_xy=True,
)


def run_benchmark(n: int):
    random.seed(42)

    xs = [random.uniform(0, 1300000) for _ in range(n)]
    ys = [random.uniform(6000000, 7200000) for _ in range(n)]

    df = pl.DataFrame({"x": xs, "y": ys})

    start = time.perf_counter()
    df.with_columns(*lambert93_to_wgs84("x", "y"))
    polars_time = time.perf_counter() - start

    start = time.perf_counter()
    REFERENCE_TRANSFORMER.transform(xs, ys)
    pyproj_time = time.perf_counter() - start

    print("=" * 60)
    print(f"Rows                : {n:,}")
    print(f"Polars              : {polars_time:.6f}s  ({n / polars_time:,.0f} rows/sec)")
    print(f"PyProj              : {pyproj_time:.6f}s  ({n / pyproj_time:,.0f} rows/sec)")
    print(f"Speed ratio         : {pyproj_time / polars_time:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        run_benchmark(n)
