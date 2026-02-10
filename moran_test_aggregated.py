#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Queen, KNN, lag_spatial
from esda.moran import Moran
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 0) Silence noisy contiguity warnings (you already handle islands explicitly)
# -----------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="libpysal.weights.contiguity"
)


# -----------------------------------------------------------------------------
# Helper: clean a GeoDataFrame + column to ensure x and weights align
# -----------------------------------------------------------------------------
def _clean_gdf_and_x(gdf: gpd.GeoDataFrame, col: str) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """
    Returns cleaned gdf (reset index, finite values only) and x as numpy array.
    """
    gdf = gdf.reset_index(drop=True)

    # Ensure numeric
    x = gdf[col].astype(float).to_numpy()

    # Keep only finite
    mask = np.isfinite(x)
    gdf = gdf.loc[mask].reset_index(drop=True)
    x = gdf[col].astype(float).to_numpy()

    # Guard: constant vector
    if np.nanvar(x) == 0:
        raise ValueError(f"{col} has zero variance after cleaning (constant values).")

    return gdf, x


# -----------------------------------------------------------------------------
# 1) Moran with Queen contiguity (drops islands, rebuilds weights)
# -----------------------------------------------------------------------------
def moran_queen_clean(
    gdf: gpd.GeoDataFrame,
    col: str,
    permutations: int = 999
) -> dict:
    """
    Global Moran's I using Queen contiguity.
    - Builds Queen weights
    - Drops islands (no neighbors)
    - Drops non-finite x values
    - Rebuilds weights after dropping rows
    """
    gdf = gdf.reset_index(drop=True)

    # Build initial Queen weights
    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = "R"

    # Drop islands
    islands = list(getattr(w, "islands", []))
    if islands:
        gdf = gdf.drop(index=islands).reset_index(drop=True)

    # Drop NaN/inf and get x
    gdf, x = _clean_gdf_and_x(gdf, col)

    # Rebuild weights after dropping rows
    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = "R"

    # Moran
    mi = Moran(x, w, permutations=permutations)

    # Count components (connectivity)
    n_components = getattr(w, "n_components", None)

    return {
        "method": "queen",
        "variable": col,
        "n": int(mi.n),
        "I": float(mi.I),
        "p_sim": float(mi.p_sim),
        "z_sim": float(mi.z_sim),
        "EI": float(mi.EI),
        "n_islands_dropped": int(len(islands)),
        "n_components": int(n_components) if n_components is not None else None,
        "weights_transform": "R",
        "permutations": int(permutations),
    }


# -----------------------------------------------------------------------------
# 2) Moran robustness with KNN (guarantees neighbors for all units)
# -----------------------------------------------------------------------------
def moran_knn_clean(
    gdf: gpd.GeoDataFrame,
    col: str,
    k: int = 8,
    permutations: int = 999
) -> dict:
    """
    Global Moran's I using KNN weights on centroids.
    - Drops non-finite values for x
    - Builds KNN weights (each unit has k neighbors)
    """
    # Clean x first (also resets index)
    gdf, x = _clean_gdf_and_x(gdf, col)

    # Use centroids for neighbor graph
    # (If you get warnings about geographic CRS, consider projecting first.)
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])

    n = len(gdf)
    if n < 3:
        raise ValueError("At least 3 observations are required for Moran's I.")

    k_eff = max(1, min(k, n - 1))

    w = KNN.from_array(coords, k=k_eff)
    w.transform = "R"

    mi = Moran(x, w, permutations=permutations)

    n_components = getattr(w, "n_components", None)

    return {
        "method": f"knn_k{k_eff}",
        "variable": col,
        "n": int(mi.n),
        "I": float(mi.I),
        "p_sim": float(mi.p_sim),
        "z_sim": float(mi.z_sim),
        "EI": float(mi.EI),
        "n_islands_dropped": 0,
        "n_components": int(n_components) if n_components is not None else None,
        "weights_transform": "R",
        "permutations": int(permutations),
        "k": int(k_eff),
    }
# -----------------------------------------------------------------------------
# 4) plot Monran I.
# -----------------------------------------------------------------------------



def moran_scatter_ecoregions(gdf, col, title):
    # --- preparar datos (mismo cleaning que antes) ---
    gdf = gdf.reset_index(drop=True)

    w = Queen.from_dataframe(gdf, use_index=False)
    islands = list(getattr(w, "islands", []))
    if islands:
        gdf = gdf.drop(index=islands).reset_index(drop=True)

    x = gdf[col].astype(float).to_numpy()
    mask = np.isfinite(x)
    gdf = gdf.loc[mask].reset_index(drop=True)
    x = gdf[col].astype(float).to_numpy()

    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = "R"

    # --- Moran ---
    mi = Moran(x, w)

    # --- scatterplot ---
    plt.figure(figsize=(6, 6))
    # wz = np.asarray(mi.wz)
    wz = np.asarray(getattr(mi, "wz", lag_spatial(w, mi.z)))
    plt.scatter(mi.z, wz, s=30, alpha=0.7, edgecolor="k")
    plt.axhline(0, color="grey", linewidth=1)
    plt.axvline(0, color="grey", linewidth=1)

    # línea de regresión (pendiente = Moran's I)
    z = np.linspace(mi.z.min(), mi.z.max(), 100)
    plt.plot(z, mi.I * z, color="red", linewidth=2,
             label=f"Moran's I = {mi.I:.2f}")

    plt.xlabel(f"Standardized {col}")
    plt.ylabel(f"Spatial lag of {col}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



# -----------------------------------------------------------------------------
# 5) Main runner: compute + export
# -----------------------------------------------------------------------------
def run():
    # Inputs
    gpkg_path = "outputs/geopackages/critical_ecoregions2.gpkg"
    layer_name = "critical_ecoregions"

    variables = ["EWMnino", "EWMnina"]
    permutations = 999
    knn_k = 8

    # Outputs
    out_dir = "outputs/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "moran_results_ecoregions.csv")
    out_json = os.path.join(out_dir, "moran_results_ecoregions.json")

    # Load data
    ecoregions_gdf = gpd.read_file(gpkg_path, layer=layer_name)

    results = []

    # Compute Moran for each variable, both weight schemes
    for col in variables:
        res_q = moran_queen_clean(ecoregions_gdf, col, permutations=permutations)
        res_k = moran_knn_clean(ecoregions_gdf, col, k=knn_k, permutations=permutations)

        results.append(res_q)
        results.append(res_k)

        # Console summary (clean)
        print(f"\n{col} — Queen:  I={res_q['I']:.3f}, p={res_q['p_sim']:.3g}, z={res_q['z_sim']:.2f}, n={res_q['n']}")
        print(f"{col} — KNN k={knn_k}: I={res_k['I']:.3f}, p={res_k['p_sim']:.3g}, z={res_k['z_sim']:.2f}, n={res_k['n']}")

    # Export
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(" -", out_csv)
    print(" -", out_json)

    # print("\nMethods paragraph:")
    # print(methods_paragraph(permutations=permutations, k=knn_k))
    moran_scatter_ecoregions(
        ecoregions_gdf,
        "EWMnino",
        "Moran scatterplot – EWMnino (ecoregions)"
    )

    moran_scatter_ecoregions(
        ecoregions_gdf,
        "EWMnina",
        "Moran scatterplot – EWMnina (ecoregions)"
    )



if __name__ == "__main__":
    run()
