import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import fiona
import os
import mapclassify



def plot_2x2_geopackage_data(path_gpkg, layer_name, border_layer):
    """Plots a 2x2 grid with specified columns from a GeoPackage."""
    # Load the data
    gdf = gpd.read_file(path_gpkg, layer=layer_name)
    borders = gpd.read_file(path_gpkg, layer=border_layer)
    
    # Define columns to plot with their respective color ramps
    columns = [
        ("EWMnino", "EWM Nino Median", "OrRd", (0, 1)),
        ("EWMnina","EWM Nina Median", "Blues", (0, 1)),
        ("primaryLoss_rate", "Tree Cover Loss Velocity (%/year)", "Greys", None),
        ("PrimaryLoss_Fires50%", "Tree Cover Loss by fires (%)", "Greys", None)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (col, title, cmap, bounds) in enumerate(columns):
        if col in gdf.columns:
            if bounds:
                scheme = mapclassify.EqualInterval(gdf[col], k=5)
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Define equal intervals of 0.2
                gdf.plot(column=col, cmap=cmap, ax=axes[i], legend=True, scheme='user_defined', classification_kwds={'bins': bins})
            else:
                gdf.plot(column=col, cmap=cmap, ax=axes[i], legend=True)
            
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
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"
    layer_name = "zonal_statistics_v2"
    borders_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"  # Replace with actual path
    borders_layer = "Neotropic_Realm"  # Replace with actual layer name
    plot_2x2_geopackage_data(path_gpkg, layer_name, borders_layer)

if __name__ == "__main__":
    run()