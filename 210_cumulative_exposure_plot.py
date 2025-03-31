import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import fiona
import os
import mapclassify
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_ewm_maps(path_gpkg, layer_name, border_layer):
    """Plots EWMnino and EWMnina with a shared color scale and individual colorbars."""
    # Load the data
    gdf = gpd.read_file(path_gpkg, layer=layer_name)
    borders = gpd.read_file(path_gpkg, layer=border_layer)
    
    columns = [
        ("EWMnino", "EWM Niño Median", "OrRd"),
        ("EWMnina", "EWM Niña Median", "Blues")
    ]

    # Calculate global min/max for shared color scale
    values_combined = pd.concat([gdf[col] for col, _, _ in columns if col in gdf.columns])
    global_min = values_combined.min()
    global_max = values_combined.max()
    norm = mpl.colors.Normalize(vmin=global_min, vmax=global_max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    labels = ['(a)', '(b)']
    
    for i, (col, title, cmap_name) in enumerate(columns):
        ax = axes[i]
        if col in gdf.columns:
            cmap = plt.get_cmap(cmap_name)

            # Plot data with shared normalization
            gdf.plot(column=col, cmap=cmap, norm=norm, ax=ax)

            # Overlay borders
            borders.plot(ax=ax, edgecolor='black', linewidth=0.1, facecolor='none')

            # Add colorbar
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cbar.ax.tick_params(labelsize=10)
        else:
            ax.set_title(f"{title} (Data Missing)")

        ax.set_title(title)
        ax.axis("off")
        # Add (a), (b) labels in corner
        ax.text(0.01, 0.95, labels[i], transform=ax.transAxes,
                fontsize=17, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("outputs/figures/ewm_maps.png", dpi=300)
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes = axes.flatten()
    
    for i, (col, title) in enumerate(columns):
        ax = axes[i]
        if col in gdf.columns:
            cmap = plt.get_cmap("Greys")
            values = gdf[col]

            # Normalize the color range
            norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())

            # Plot data
            gdf.plot(column=col, cmap=cmap, norm=norm, ax=ax)

            # Add borders
            borders.plot(ax=ax, edgecolor='black', linewidth=0.1, facecolor='none')

            # Add colorbar with smaller size
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []  # Required for ScalarMappable to work without data
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)  # <--- Control size with shrink
            cbar.ax.tick_params(labelsize=8)  # Optional: smaller tick font
        else:
            ax.set_title(f"{title} (Data Missing)")

        ax.set_title(title)
        ax.axis("off")
    
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
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, (col, title) in enumerate(columns):
        ax = axes[i]
        if col in gdf.columns:
            cmap = plt.get_cmap("Greys")  # Moved inside the if block
            values = gdf[col]

            # Normalize the color range
            norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())

            # Plot data
            gdf.plot(column=col, cmap=cmap, norm=norm, ax=ax)

            # Add borders
            borders.plot(ax=ax, edgecolor='black', linewidth=0.1, facecolor='none')

            # Add colorbar with smaller size
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
            cbar.ax.tick_params(labelsize=8)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
        else:
            ax.set_title(f"{title} (Data Missing)")

        ax.set_title(title)
        ax.axis("off")
        labels = ['(c)', '(d)', '(e)']
        ax.text(0.01, 0.95, labels[i], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    
    plt.tight_layout()
    plt.savefig("outputs/figures/lossforest_drivers.png", dpi=300, bbox_inches="tight")

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
    plot_ewm_maps(path_gpkg, layer_name, borders_layer)


if __name__ == "__main__":
    run()