import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import fiona
import os
import mapclassify

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd

def plot_custom_2row_map(path_gpkg, layer_name, border_layer):
    gdf = gpd.read_file(path_gpkg, layer=layer_name)
    borders = gpd.read_file(path_gpkg, layer=border_layer)

    columns = [
        ("EWMnino", "EWM Nino Median", "OrRd", (0, 1)),
        ("EWMnina", "EWM Nina Median", "Blues", (0, 1)),
        ("PrimaryLoss_Rate50", "Tree Cover Loss Velocity (%/year)", "Greys", None),
        ("proportion_fire-induced_Primaryloss", "Tree Cover Loss by Fires (%)", "Greys", None),
        ("PrimaryForest_Loss50", "Cumulative Primary Forest Loss (ha)", "Greys", None)
    ]
    fig = plt.figure(figsize=(15, 10))
    spec = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1, 1])

    # First row: center by spanning the middle two columns
    ax0 = fig.add_subplot(spec[0, 1])
    ax1 = fig.add_subplot(spec[0, 2])

    # Second row: fixed 3 columns
    ax2 = fig.add_subplot(spec[1, 0])
    ax3 = fig.add_subplot(spec[1, 1])
    ax4 = fig.add_subplot(spec[1, 2])

    axes = [ax0, ax1, ax2, ax3, ax4]

    for i, (col, title, cmap, bounds) in enumerate(columns):
        ax = axes[i]
        if col in gdf.columns:
            if bounds:
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                gdf.plot(column=col, cmap=cmap, ax=ax, legend=True,
                         scheme='user_defined', classification_kwds={'bins': bins})
            else:
                gdf.plot(column=col, cmap=cmap, ax=ax, legend=True)

            borders.plot(ax=ax, edgecolor='black', linewidth=0.1, facecolor='none')
            ax.set_title(title)
            ax.axis("off")
        else:
            ax.set_title(f"{title} (Data Missing)")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/figures/custom_2row_map.png", dpi=300)
    plt.show()
    
def plot_2x3_geopackage_data(path_gpkg, layer_name, border_layer):
    gdf = gpd.read_file(path_gpkg, layer=layer_name)
    borders = gpd.read_file(path_gpkg, layer=border_layer)

    columns = [
        ("EWMnino", "EWM Nino Median", "OrRd", (0, 1)),
        ("EWMnina", "EWM Nina Median", "Blues", (0, 1)),
        ("PrimaryLoss_Rate50", "Tree Cover Loss Velocity (%/year)", "Greys", None),
        ("proportion_fire-induced_Primaryloss", "Tree Cover Loss by Fires (%)", "Greys", None),
        ("PrimaryForest_Loss50", "Cumulative Primary Forest Loss (ha)", "YlOrBr", None)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (col, title, cmap, bounds) in enumerate(columns):
        if col in gdf.columns:
            if bounds:
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                gdf.plot(column=col, cmap=cmap, ax=axes[i], legend=True,
                         scheme='user_defined', classification_kwds={'bins': bins})
            else:
                gdf.plot(column=col, cmap=cmap, ax=axes[i], legend=True)

            borders.plot(ax=axes[i], edgecolor='black', linewidth=0.1, facecolor='none')
            axes[i].set_title(title)
            axes[i].axis("off")
        else:
            axes[i].set_title(f"{title} (Data Missing)")
            axes[i].axis("off")

    # Remove any unused axes
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_2x2_geopackage_data(path_gpkg, layer_name, border_layer):
    """Plots a 2x2 grid with specified columns from a GeoPackage."""
    # Load the data
    gdf = gpd.read_file(path_gpkg, layer=layer_name)
    borders = gpd.read_file(path_gpkg, layer=border_layer)
    
    # Define columns to plot with their respective color ramps
    # columns = [
    #     ("EWMnino", "EWM Nino Median", "OrRd", (0, 1)),
    #     ("EWMnina","EWM Nina Median", "Blues", (0, 1)),
    #     ("primaryLoss_rate", "Tree Cover Loss Velocity (%/year)", "Greys", None),
    #     ("PrimaryLoss_Fires50%", "Tree Cover Loss by fires (%)", "Greys", None)
    # ]
    columns = [
        # ("EWMnino", "EWM Nino Median", "OrRd", (0, 1)),
        # ("EWMnina", "EWM Nina Median", "Blues", (0, 1)),
        ("PrimaryLoss_Rate50", "Tree Cover Loss Velocity (%/year)"),
        ("proportion_fire-induced_Primaryloss", "Tree Cover Loss by Fires (%)"),
        ("PrimaryForest_Loss50", "Cumulative Primary Forest Loss (ha)")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (col, title) in enumerate(columns):
        if col in gdf.columns:
            gdf.plot(column=col, cmap="Greys", ax=axes[i], legend=True)
            
            borders.plot(ax=axes[i], edgecolor='black', linewidth=0.1, facecolor='none')
            axes[i].set_title(title)
            axes[i].axis("off")
        else:
            # Overlay the borders layer
            axes[i].set_title(f"{title} (Data Missing)")
            axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()


def run():
    # Example usage
    # path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"
    # layer_name = "zonal_statistics_v2"
    borders_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"  # Replace with actual path
    borders_layer = "Neotropic_Realm"  # Replace with actual layer name
    # plot_2x2_geopackage_data(path_gpkg, layer_name, borders_layer)

        # Example usage
    # path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"
    # layer_name = "zonal_statistics_v2"
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"
    layer_name = "ZonalStat_Ecoregions"
    borders_layer = "Neotropical"  # Replace with actual layer name
    gdf = gpd.read_file(path_gpkg, layer=layer_name)
    print(gdf.dtypes)
    # plot_custom_2row_map(path_gpkg, layer_name, borders_layer)
    plot_2x2_geopackage_data(path_gpkg, layer_name, borders_layer)


if __name__ == "__main__":
    run()