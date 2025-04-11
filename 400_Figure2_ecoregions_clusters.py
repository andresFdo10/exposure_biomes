import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def add_features(ax, extent  = (-105, -33.9, -31.5, 23.8)):
    # Add coastlines
    ax.coastlines()

    # Add country borders with a dotted line style
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    # Set extent for dimension standardization
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add land features with an edge color
    land = cfeature.NaturalEarthFeature("physical", "land", "50m", facecolor="none")
    ax.add_feature(land, edgecolor="black", linewidth=0.2)

    # Add gridlines with specified transparency and line width
    # ✅ Correct way to control gridline labels
    gl = ax.gridlines(alpha=0.6, linewidth=0.2, draw_labels=True)
    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = True
    # ✅ Set decimal degree format
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(-105, -30, 15))
    # gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 10))
    # gl.xformatter = mticker.FormatStrFormatter("%.1f")
    # gl.yformatter = mticker.FormatStrFormatter("%.1f")
    gl.xlabel_style = {'size': 9, 'color': 'black'}
    gl.ylabel_style = {'size': 9, 'color': 'black', 'rotation': 90}
    # Rotate right-side labels and set smaller font
    # ✅ Trigger rendering to populate label artists
    plt.draw()
def run():

    # ✅ Step 1: Load the original GeoPackage with ecoregions
    gpkg_path = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"  # Update this if needed
    layer_name = "ZonalStat_Ecoregions"  # Update if your layer name is different
    ecoregions_gdf = gpd.read_file(gpkg_path, layer=layer_name)

    # ✅ Step 2: Load the PCA clusters CSV
    clusters_df = pd.read_csv("outputs/csv/pca_reduced_data_with_clusters.csv")

    # ✅ Step 3: Merge using "ECO_ID" (keeping all ecoregions)
    merged_gdf = ecoregions_gdf.merge(clusters_df, on="ECO_ID", how="left")

    # ✅ Step 4: Save as a new GeoPackage file
    output_gpkg = "outputs/geopackages/ecoregions_with_clusters.gpkg"
    merged_gdf.to_file(output_gpkg, driver="GPKG", layer="ecoregions_clusters")

    print(f"✅ Merged GeoPackage saved at: {output_gpkg}")

    # ✅ Step 1: Load the merged GeoPackage
    gpkg_path = "outputs/geopackages/ecoregions_with_clusters.gpkg"
    layer_name = "ecoregions_clusters"
    ecoregions_gdf = gpd.read_file(gpkg_path, layer=layer_name)

    # ✅ Step 2: Handle NaN Values in Cluster Column
    if "Cluster" not in ecoregions_gdf.columns:
        raise ValueError("⚠️ The 'Cluster' column is missing in the GeoPackage!")

    print(ecoregions_gdf.groupby("Cluster").size())

    # Replace NaNs with -1 (used for transparency)
    ecoregions_gdf["Cluster"].fillna(-1, inplace=True)

        # ✅ Step 1: Get unique valid clusters (excluding -1)
    valid_clusters = sorted([c for c in ecoregions_gdf["Cluster"].unique() if c != -1])
    num_clusters = len(valid_clusters)

    # ✅ Step 4: Define a Custom Colormap
    cmap = plt.get_cmap("viridis", num_clusters)  # Use a discrete colormap for clusters
    norm = mcolors.Normalize(vmin=min(valid_clusters), vmax=max(valid_clusters))

    # ✅ Step 5: Assign Colors (Make `-1` Transparent)
    color_dict = {c: cmap(norm(c)) for c in valid_clusters if c != -1}  # Normal clusters
    color_dict[-1] = (0, 0, 0, 0)  # RGBA → Transparent for `-1`

    # ✅ Step 6: Map Colors to Data
    ecoregions_gdf["color"] = ecoregions_gdf["Cluster"].map(color_dict)


    # ***** ✅ Step 7: Create a Figure ***** 
    fig, ax = plt.subplots(
        figsize=(12, 8), 
        subplot_kw={"projection": ccrs.PlateCarree()}
        )

    # ✅ Step 8: Plot the Map with Transparent `-1`
    ecoregions_gdf.plot(
        color=ecoregions_gdf["color"],  # Uses pre-defined colors
        edgecolor="black", 
        linewidth=0.05,
        ax=ax
    )

    # Define your cluster labels
    cluster_labels = {
        0: "1: Fire-Affected Forests",
        1: "2: ENSO-Exposed Regions",
        2: "3: Stable/Mixed Impact",
        3: "4: High Deforestation",
        # -1: "No Data"
        # Add more if needed
    }

    ecoregions_gdf["Cluster"] = ecoregions_gdf["Cluster"].astype(int)
    ecoregions_gdf["Cluster"] = ecoregions_gdf["Cluster"].replace(-1, np.nan)
    print(ecoregions_gdf["Cluster"].unique())

    # Add cluster legend manually
    if "Cluster" in ecoregions_gdf.columns:
        cluster_ids = sorted(ecoregions_gdf["Cluster"].dropna().unique())
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(cluster_ids), vmax=max(cluster_ids))

        handles = []
        for cid in cluster_ids:
            color = cmap(norm(cid))
            label = cluster_labels.get(cid, f"Cluster {cid}")
            patch = mpatches.Patch(color=color, label=label)
            handles.append(patch)

        ax.legend(
            handles=handles, 
            title="Clusters", 
            loc="best", 
            fontsize=10, 
            title_fontsize=11
            )

    add_features(ax, extent=(-110, -33.9, -40.5, 28.5))
    # ✅ Step 10: Finalize Plot
    # ax.set_title("Ecoregions Colored by Clusters (Transparent -1)", fontsize=16)
    # ax.set_axis_off()  # Hide axis for a cleaner look

    plt.savefig("./outputs/figures/Ecoregions_Cluster.png", dpi=300, bbox_inches='tight')

    # ✅ Step 11: Force Show Plot
    plt.show()


if __name__ == '__main__':
    run()