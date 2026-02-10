import numpy as np
import rasterio
from rasterio.transform import xy
from libpysal.weights import KNN
from esda.moran import Moran


def moran_raster_knn(tif_path, sample_n=30000, k=8, seed=42):
    rng = np.random.default_rng(seed)

    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        transform = src.transform
        nodata = src.nodata

        if arr.ndim != 2:
            raise ValueError(f"Raster {tif_path} is not 2D")

        finite = np.isfinite(arr)
        if nodata is not None:
            if np.isnan(nodata):
                valid = finite
            else:
                valid = finite & (arr != nodata)
        else:
            valid = finite

        rows, cols = np.where(valid)
        vals = arr[rows, cols].astype(float)

        if len(vals) == 0:
            raise ValueError(f"No valid pixels found in {tif_path}")

        take = min(sample_n, len(vals))
        idx = rng.choice(len(vals), size=take, replace=False)

        rows_s = rows[idx]
        cols_s = cols[idx]
        vals_s = vals[idx]

        xs, ys = xy(transform, rows_s, cols_s, offset="center")
        coords = np.column_stack([xs, ys])

    if take < 3:
        raise ValueError("At least 3 valid sampled pixels are required for Moran's I.")

    k_eff = max(1, min(k, take - 1))

    w = KNN.from_array(coords, k=k_eff)
    w.transform = "R"

    mi = Moran(vals_s, w, permutations=999)

    return {
        "n": take,
        "k": int(k_eff),
        "I": float(mi.I),
        "p_sim": float(mi.p_sim),
        "z_sim": float(mi.z_sim),
    }


def run():
    
    # EWI El Niño
    res_nino = moran_raster_knn("outputs/raster/exposure_EWM_nino.tif")
    print(res_nino)

    # EWI La Niña
    res_nina = moran_raster_knn("outputs/raster/exposure_EWM_nina.tif")
    print(res_nina)


if __name__ == "__main__":
    run()
