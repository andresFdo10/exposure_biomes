import string
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
from cartopy import crs as ccrs
from shapely.geometry import mapping
import cartopy.feature as cfeature


def clip_raster(raster_path):
    ds = rxr.open_rasterio(raster_path)
    raster = ds.squeeze()  # Remove single-dimensional coordinates
    if raster.ndim == 3:
        raster = raster.isel(band=0)  # Select the first band if multiple bands
    return raster


def run():
    """
    This function reads raster files and plots their impact drivers on a map.

    Parameters:
    None

    Returns:
    None
    """

    # Set up coordinate reference system
    crs = "epsg:4326"

    # Load mask geometry
    # path_gpkg = (
    #     "~/Nextcloud2/01_article/produced/01_GIS_Data/01_VECTOR/SuitabilityAreas.gpkg"
    # )
    # mask_geometry = gpd.read_file(
    #     path_gpkg, layer="suit_coffea_theobroma", crs=crs
    # )
    # print(f"Initial Vector CRS: {mask_geometry.crs}")

    # Clip rasters La Nina
    tmax = clip_raster("./outputs/raster/aligned_tmmx_nina_60yrs.tif")
    tmin = clip_raster("./outputs/raster/aligned_tmmn_nina_60yrs.tif")
    pr = clip_raster("./outputs/raster/aligned_pr_nina_60yrs.tif")
    pdsi = clip_raster("./outputs/raster/aligned_pdsi_nina_60yrs.tif")

    drivers_list = [tmax, tmin, pr, pdsi]

    # Plot the impact drivers
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 6))
    cmap1 = "BrBG_r"
    cmap2 = "Spectral"
    levels = np.linspace(-0.70, 0.70, 9)
    letter = ["(a)", "(b)", "(c)", "(d)"]

    raster_names = ["Tmax", "Tmin", "PR", "PDSI"]
    for i, (raster, name) in enumerate(zip(drivers_list, raster_names)):
        ax = fig.add_subplot(2, 2, i + 1, projection=projection)
        plot = raster.plot(
            ax=ax,
            cmap=cmap2 if name in ["Tmax", "Tmin"] else cmap1,
            levels=levels,
            extend="both",
        )
        cbar = plot.colorbar
        cbar.set_label("")
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_title(f"ONI and {name}")
        ax.coastlines(alpha=0.4, linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3)
        ax.gridlines(alpha=0.4, linewidth=0.3)
        # mask_geometry.plot(
        #     ax=ax, facecolor="none", edgecolor="black", alpha=0.4, linewidth=0.1
        # )
        # letter = string.ascii_lowercase[i]
        ax.text(
            0.95,
            0.95,
            letter[i],
            transform=ax.transAxes,
            # fontsize=14,
            va="top",
            ha="right",
            fontweight="bold"
        )

    plt.tight_layout()
    # fig.subplots_adjust(hspace=0.08)
    plt.savefig("./outputs/raster/climatic-impact-drivers_nina.png", dpi=600)
    plt.show()

    # -------------------------------------------------------------------------
    tmax = clip_raster("./outputs/raster/aligned_tmmx_nino_60yrs.tif")
    tmin = clip_raster("./outputs/raster/aligned_tmmn_nino_60yrs.tif")
    pr = clip_raster("./outputs/raster/aligned_pr_nino_60yrs.tif")
    pdsi = clip_raster("./outputs/raster/aligned_pdsi_nino_60yrs.tif")

    drivers_list = [tmax, tmin, pr, pdsi]

    # Plot the impact drivers
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 6))
    cmap1 = "BrBG"
    cmap2 = "Spectral_r"
    levels = np.linspace(-0.70, 0.70, 9)

    raster_names = ["Tmax", "Tmin", "PR", "PDSI"]
    for i, (raster, name) in enumerate(zip(drivers_list, raster_names)):
        ax = fig.add_subplot(2, 2, i + 1, projection=projection)
        plot = raster.plot(
            ax=ax,
            cmap=cmap2 if name in ["Tmax", "Tmin"] else cmap1,
            levels=levels,
            extend="both",
        )
        cbar = plot.colorbar
        cbar.set_label("")
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_title(f"ONI and {name}")
        ax.coastlines(alpha=0.4, linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3)
        ax.gridlines(alpha=0.4, linewidth=0.3)
        # mask_geometry.plot(
        #     ax=ax, facecolor="none", edgecolor="black", alpha=0.4, linewidth=0.1
        # )
        # letter = string.ascii_lowercase[i]
        ax.text(
            0.95,
            0.95,
            letter[i],
            transform=ax.transAxes,
            # fontsize=14,
            va="top",
            ha="right",
            fontweight="bold"
        )

    plt.tight_layout()
    # fig.subplots_adjust(hspace=0.08)
    plt.savefig("./outputs/raster/climatic-impact-drivers_nino.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    run()
