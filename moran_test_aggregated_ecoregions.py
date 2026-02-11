#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from libpysal.weights import DistanceBand
from esda.moran import Moran


# ---------------------------------------------------------------------
# Prep: read gpkg, clean values, project to metric CRS, build centroids
# ---------------------------------------------------------------------
def prepare_ecoregion_centroids(gpkg_path, layer_name, value_col, target_epsg=3857):
    gdf = gpd.read_file(gpkg_path, layer=layer_name)

    # Keep finite values
    x = gdf[value_col].astype(float).to_numpy()
    mask = np.isfinite(x)
    gdf = gdf.loc[mask].copy().reset_index(drop=True)

    if gdf.crs is None:
        raise ValueError(
            "GeoPackage CRS is None. Please ensure it has a valid CRS (e.g., EPSG:4326)."
        )

    # Project to meters for distance-based weights
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=target_epsg)

    # Use centroids in meters
    cent = gdf.geometry.centroid
    coords = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    values = gdf[value_col].astype(float).to_numpy()

    # Guard: constant
    if np.nanvar(values) == 0:
        raise ValueError(f"{value_col} has zero variance after cleaning.")

    return coords, values, gdf.crs


# ---------------------------------------------------------------------
# Moran correlogram with permutation bands
# ---------------------------------------------------------------------
def moran_correlogram_with_bands(coords, values, distances_m, permutations=199):
    I_obs, pvals = [], []
    ci_lo, ci_hi = [], []
    env_lo, env_hi = [], []

    for d in distances_m:
        w = DistanceBand.from_array(coords, threshold=float(d), binary=True, silence_warnings=True)
        w.transform = "R"

        mi = Moran(values, w, permutations=permutations)

        sims = np.asarray(mi.sim)  # permutation distribution of Moran's I

        I_obs.append(mi.I)
        pvals.append(mi.p_sim)

        # 95% CI under null
        ci_lo.append(np.quantile(sims, 0.025))
        ci_hi.append(np.quantile(sims, 0.975))

        # envelope under null
        env_lo.append(np.min(sims))
        env_hi.append(np.max(sims))

    return (np.asarray(I_obs), np.asarray(pvals),
            np.asarray(ci_lo), np.asarray(ci_hi),
            np.asarray(env_lo), np.asarray(env_hi))


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def plot_correlogram_bands(dist_km, I_obs, ci_lo, ci_hi, env_lo, env_hi, title):
    plt.figure(figsize=(7.5, 4))

    # Permutation envelope (min-max)
    plt.fill_between(dist_km, env_lo, env_hi, alpha=0.20, label="Max. Moran's I (perm envelope)")

    # 95% CI (null)
    plt.fill_between(dist_km, ci_lo, ci_hi, alpha=0.30, label="Conf. Interv. (95%)")

    # Observed line
    plt.plot(dist_km, I_obs, marker="o", linewidth=1.7, label="Moran's I")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Distance (km)")
    plt.ylabel("Moran's I")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run():
    gpkg_path = "outputs/geopackages/critical_ecoregions2.gpkg"
    layer_name = "critical_ecoregions"

    # Distances in meters (converted to km for plotting)
    distances_m = np.arange(100_000, 2_600_000, 200_000)  # 100 km to 2500 km
    dist_km = distances_m / 1000

    permutations = 999  # use 999 for the final supplementary figure

    # -------------------------
    # 1) El Niño
    # -------------------------
    coords, values, crs = prepare_ecoregion_centroids(gpkg_path, layer_name, "EWMnino")
    print("Projected CRS used:", crs)

    I_obs, pvals, ci_lo, ci_hi, env_lo, env_hi = moran_correlogram_with_bands(
        coords, values, distances_m, permutations=permutations
    )

    plot_correlogram_bands(
        dist_km, I_obs, ci_lo, ci_hi, env_lo, env_hi,
        title="Moran correlogram – El Niño (EWMnino) | Ecoregions"
    )

    # -------------------------
    # 2) La Niña
    # -------------------------
    coords, values, crs = prepare_ecoregion_centroids(gpkg_path, layer_name, "EWMnina")

    I_obs, pvals, ci_lo, ci_hi, env_lo, env_hi = moran_correlogram_with_bands(
        coords, values, distances_m, permutations=permutations
    )

    plot_correlogram_bands(
        dist_km, I_obs, ci_lo, ci_hi, env_lo, env_hi,
        title="Moran correlogram – La Niña (EWMnina) | Ecoregions"
    )


if __name__ == "__main__":
    run()


