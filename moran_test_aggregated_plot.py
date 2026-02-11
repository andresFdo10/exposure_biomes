#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rasterio
from rasterio.transform import xy
from libpysal.weights import DistanceBand
from esda.moran import Moran
import matplotlib.pyplot as plt
from pyproj import Transformer


def sample_pixels_from_raster(tif_path, sample_n=10000, seed=42):
    rng = np.random.default_rng(seed)

    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

        if nodata is not None:
            valid = arr != nodata
        else:
            valid = ~np.isnan(arr)

        rows, cols = np.where(valid)
        vals = arr[rows, cols].astype(float)

        if len(vals) == 0:
            raise ValueError(f"No valid pixels in {tif_path}")

        take = min(sample_n, len(vals))
        idx = rng.choice(len(vals), size=take, replace=False)

        rows_s = rows[idx]
        cols_s = cols[idx]
        vals_s = vals[idx]

        xs, ys = xy(transform, rows_s, cols_s, offset="center")
        coords = np.column_stack([xs, ys])

    return coords, vals_s, crs


def moran_correlogram(coords, values, distances, permutations=199):
    I_obs = []
    ci_lo, ci_hi = [], []
    env_lo, env_hi = [], []

    for d in distances:
        w = DistanceBand.from_array(coords, threshold=float(d), binary=True, silence_warnings=True)
        w.transform = "R"

        mi = Moran(values, w, permutations=permutations)
        sims = np.asarray(mi.sim)

        I_obs.append(mi.I)
        ci_lo.append(np.quantile(sims, 0.025))
        ci_hi.append(np.quantile(sims, 0.975))
        env_lo.append(np.min(sims))
        env_hi.append(np.max(sims))

    return (np.asarray(I_obs),
            np.asarray(ci_lo), np.asarray(ci_hi),
            np.asarray(env_lo), np.asarray(env_hi))


def plot_moran_correlogram(distances, I_obs, ci_lo, ci_hi, env_lo, env_hi, title, xlabel):
    plt.figure(figsize=(7.5, 4))

    plt.fill_between(distances, env_lo, env_hi, alpha=0.25, label="Max. Moran's I (perm envelope)")
    plt.fill_between(distances, ci_lo, ci_hi, alpha=0.35, label="Conf. Interv. (95%)")
    plt.plot(distances, I_obs, marker="o", linewidth=1.5, label="Moran's I")

    plt.axhline(0, linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("Moran's I")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def lonlat_to_3857(coords_lonlat):
    """Assume coords are lon/lat degrees and project to meters (EPSG:3857)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_m, y_m = transformer.transform(coords_lonlat[:, 0], coords_lonlat[:, 1])
    return np.column_stack([x_m, y_m])


def run():
    tif_nino = "outputs/raster/exposure_EWM_nino.tif"
    tif_nina = "outputs/raster/exposure_EWM_nina.tif"

    sample_n = 2000
    permutations = 499  # use 999 for final

    # Distances in meters (because we will project coords to EPSG:3857)
    distances = np.arange(20000, 520000, 20000)  # 20 km to 500 km

    # ---- El Niño ----
    coords, vals, crs = sample_pixels_from_raster(tif_nino, sample_n=sample_n, seed=42)
    print("El Niño CRS (from raster):", crs)

    # IMPORTANT: raster CRS is None, but coords are lon/lat -> project to meters
    coords_m = lonlat_to_3857(coords)

    I_obs, ci_lo, ci_hi, env_lo, env_hi = moran_correlogram(coords_m, vals, distances, permutations=permutations)
    plot_moran_correlogram(distances, I_obs, ci_lo, ci_hi, env_lo, env_hi,
                           title="Moran correlogram – El Niño EWI (pixel sample)",
                           xlabel="Distance (meters)")

    # ---- La Niña ----
    coords, vals, crs = sample_pixels_from_raster(tif_nina, sample_n=sample_n, seed=43)
    print("La Niña CRS (from raster):", crs)

    coords_m = lonlat_to_3857(coords)

    I_obs, ci_lo, ci_hi, env_lo, env_hi = moran_correlogram(coords_m, vals, distances, permutations=permutations)
    plot_moran_correlogram(distances, I_obs, ci_lo, ci_hi, env_lo, env_hi,
                           title="Moran correlogram – La Niña EWI (pixel sample)",
                           xlabel="Distance (meters)")


if __name__ == "__main__":
    run()
