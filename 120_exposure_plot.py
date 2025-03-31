import string
import rioxarray as rxr
# import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as xrx
import xarray as xr
from cartopy import crs as ccrs
import cartopy.feature as cfeature



def plot_exposure(ax, exposure_data, title, letter, cmap, extent):
    """
    Plots the exposure data on a given axis with additional vector overlays.
    
    Parameters:
    - ax: Matplotlib axis
    - exposure_data: The exposure raster data
    - vector_data: The mask geometry for dimension standardization
    - overlay_data: The additional vector layer (Coffea or Theobroma)
    - title: Title for the subplot
    - letter: Letter identifier for the subplot
    - cmap: Colormap to use for the raster plot
    """
    projection = ccrs.PlateCarree()

    # Plot raster data
    img = exposure_data.plot(
        ax=ax,
        add_colorbar=False,
        cmap=cmap,
        transform=projection,
    )

    # Plot vector overlay (Coffea or Theobroma)
    # overlay_data.plot(
    #     ax=ax, 
    #     facecolor="none",
    #     edgecolor="black",
    #     linewidth=0.01,
    #     alpha = 0.5,
    #     transform=projection
    # )

    # Set extent for dimension standardization
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add title, coastlines, and gridlines
    ax.set_title(title, fontsize=12)
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    land = cfeature.NaturalEarthFeature("physical", "land", "50m", facecolor="none")
    ax.gridlines(linewidth=0.3, draw_labels=False)
    ax.text(
        0.04,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        va="top",
        ha="right",
        fontweight="bold",
    )

    return img



def run():
    # Set up coordinate reference system
    crs = "epsg:4326"

    # Load mask geometry
    # path_gpkg = (
    #     "~/Nextcloud2/01_article/produced/01_GIS_Data/01_VECTOR/SuitabilityAreas.gpkg"
    # )
    # path_gpkg1 = (
    #     "~/Nextcloud2/01_article/produced/01_GIS_Data/01_VECTOR/01_VECTORIAL_V3.gpkg"
    # )
    # mask_geometry = gpd.read_file(
    #     path_gpkg1, layer="SuitabilityAreas_CoffeaA_TheobromaC_WGS84", crs=crs
    # )
    # theobroma = gpd.read_file(path_gpkg, layer="theobroma")
    # coffea = gpd.read_file(path_gpkg, layer="coffea")


        # Load entropy-weighted exposure maps
    exposure_maps = {
        "exp_el_nino": rxr.open_rasterio("./outputs/raster/exposure_EWM_nino.tif", masked=True).where(lambda x: x != 0),
        "exp_la_nina": rxr.open_rasterio("./outputs/raster/exposure_EWM_nina.tif", masked=True).where(lambda x: x != 0),
    }

    # Plotting
    cmap1 = "YlOrBr"  # El Niño colormap
    cmap2 = "YlGnBu"  # La Niña colormap
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 10),
        sharex=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
        # constrained_layout=True
        )

    # Titles, letters, overlays, and colormaps for subplots
    titles = [
        "Exposure El Niño",
        "Exposure La Niña",
    ]
    letters = list(string.ascii_lowercase)[:2]
    cmaps = [cmap1, cmap2]  # Match colormap to event type
    # overlays = [coffea, coffea, theobroma, theobroma]
    # regions = ["coffea", "coffea", "theobroma", "theobroma"]
    events = ["el_nino", "la_nina"]
    # extent = (-104.8, -34.9, -29.5, 23.5)
    extent = (-105, -33.9, -31.5, 23.8)

    imgs = []
    for i, ax in enumerate(ax.flat):
        exposure_key = f"exp_{events[i]}"
        img = plot_exposure(
            ax,
            exposure_maps[exposure_key][0],  # Extract first layer if multi-band
            # mask_geometry,
            # overlays[i],
            titles[i],
            letters[i],
            cmaps[i],
            extent
        )
        imgs.append(img)

    # Add colorbar for each colormap
    cbar_ax1 = fig.add_axes([0.1, 0.25, 0.35, 0.02])  # Position for El Niño colorbar
    cbar_ax2 = fig.add_axes([0.55, 0.25, 0.35, 0.02])  # Position for La Niña colorbar

    cbar1 = plt.colorbar(imgs[0], cax=cbar_ax1, orientation="horizontal", cmap=cmap1)
    cbar2 = plt.colorbar(imgs[1], cax=cbar_ax2, orientation="horizontal", cmap=cmap2)

    cbar1.set_label("Exposure Index - El Niño", fontsize=12)
    cbar2.set_label("Exposure Index - La Niña", fontsize=12)
    cbar1.ax.tick_params(labelsize=11)
    cbar2.ax.tick_params(labelsize=11)

    plt.tight_layout()  # Adjust layout to fit the colorbars
    fig.subplots_adjust(hspace=0.08)
    plt.savefig("./outputs/figures/Exposure_Entropy_El_Nino_La_Nina.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run()