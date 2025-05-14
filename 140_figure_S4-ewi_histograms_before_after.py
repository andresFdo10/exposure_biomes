import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
def run():
    # Load rasters
    ewi_nino_60yrs = rxr.open_rasterio("./outputs/raster/exposure_EWM_nino_60yrs.tif")
    ewi_nina_60yrs = rxr.open_rasterio("./outputs/raster/exposure_EWM_nina_60yrs.tif")
    ewi_nino_after = rxr.open_rasterio("./outputs/raster/exposure_EWM_nino_after.tif")
    ewi_nina_after = rxr.open_rasterio("./outputs/raster/exposure_EWM_nina_after.tif")
    ewi_nino_before = rxr.open_rasterio("./outputs/raster/exposure_EWM_nino_before.tif")
    ewi_nina_before = rxr.open_rasterio("./outputs/raster/exposure_EWM_nina_before.tif")

    # Flatten and drop NaN values
    ewi_nino_60yrs = ewi_nino_60yrs.where(ewi_nino_60yrs != 0)
    ewi_nino_60yrs_flat = ewi_nino_60yrs.values.flatten()
    ewi_nino_60yrs_flat = ewi_nino_60yrs_flat[~np.isnan(ewi_nino_60yrs_flat)]

    ewi_nina_60yrs = ewi_nina_60yrs.where(ewi_nina_60yrs != 0)
    ewi_nina_60yrs_flat = ewi_nina_60yrs.values.flatten()
    ewi_nina_60yrs_flat = ewi_nina_60yrs_flat[~np.isnan(ewi_nina_60yrs_flat)]

    ewi_nino_after = ewi_nino_after.where(ewi_nino_after != 0)
    ewi_nino_after_flat = ewi_nino_after.values.flatten()
    ewi_nino_after_flat = ewi_nino_after_flat[~np.isnan(ewi_nino_after_flat)]

    ewi_nina_after = ewi_nina_after.where(ewi_nina_after != 0)
    ewi_nina_after_flat = ewi_nina_after.values.flatten()
    ewi_nina_after_flat = ewi_nina_after_flat[~np.isnan(ewi_nina_after_flat)]

    ewi_nino_before = ewi_nino_before.where(ewi_nino_before != 0)
    ewi_nino_before_flat = ewi_nino_before.values.flatten()
    ewi_nino_before_flat = ewi_nino_before_flat[~np.isnan(ewi_nino_before_flat)]

    ewi_nina_before = ewi_nina_before.where(ewi_nina_before != 0)
    ewi_nina_before_flat = ewi_nina_before.values.flatten()
    ewi_nina_before_flat = ewi_nina_before_flat[~np.isnan(ewi_nina_before_flat)]


    titles = ["EWM Niño 1961-1990", "EWM Niño 1991-2020", "EWM Niño 1961-2021", "EWM Niña 1961-1990", "EWM Niña 1991-2020", "EWM Niña 1961-2021"]
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    data = [ewi_nino_before_flat, ewi_nino_after_flat, ewi_nino_60yrs_flat, ewi_nina_before_flat, ewi_nina_after_flat, ewi_nina_60yrs_flat]
    fig, axs = plt.subplots(2, 3, figsize=(11, 8), sharey='row')
    for i, ax in enumerate(axs.flat):
        if titles[i] in ["EWM Niño 1961-1990", "EWM Niño 1991-2021", "EWM Niño 1961-2021"]:
            ax.set_ylim(0, 150000)
            ax.hist(data[i], bins=20, alpha=0.5)
        else:
            ax.set_ylim(0, 170000)
            ax.hist(data[i], bins=20, alpha=0.5)

        ax.text(0.90, 1.02, letters[i], transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.set_title(titles[i])
        ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig("./outputs/figures/Figure_S4_ewi_standard_deviation_by_pixel.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run()