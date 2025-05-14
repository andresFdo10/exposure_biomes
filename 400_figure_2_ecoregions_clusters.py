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
from matplotlib.colors import ListedColormap

def add_features(ax, extent  = (-105, -33.9, -31.5, 23.8)):
    # Add coastlines
    ax.coastlines()

    # Add country borders with a dotted line style
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    # Set extent for dimension standardization
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add land features with an edge color
    land = cfeature.NaturalEarthFeature(
        "physical", 
        "land", 
        "50m", 
        facecolor="none"
        )
    ax.add_feature(
        land, 
        edgecolor="black", 
        linewidth=0.2
        )

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
    """
    Main function to run the clustering analysis and generate the map with ecoregions
    colored by their respective clusters.

    This function performs the following steps:

    1. Loads the original GeoPackage with ecoregions.
    2. Loads the PCA clusters CSV.
    3. Merges the two datasets using the "ECO_ID" column.
    4. Saves the merged dataset as a new GeoPackage file.
    5. Loads the merged GeoPackage.
    6. Plots the map with ecoregions colored by their respective clusters.
    7. Adds a legend with cluster labels.
    8. Saves the plot as a PNG file.
    """
    # Step 1: Load the original GeoPackage with ecoregions
    gpkg_path = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"
    layer_name = "ZonalStat_Ecoregions"
    ecoregions_gdf = gpd.read_file(gpkg_path, layer=layer_name)

    # Step 2: Load the PCA clusters CSV
    clusters_df = pd.read_csv("outputs/csv/pca_reduced_data_with_clusters.csv")

    # Step 3: Merge using "ECO_ID" (keeping all ecoregions)
    merged_gdf = ecoregions_gdf.merge(clusters_df, on="ECO_ID", how="left")

    # Step 4: Save as a new GeoPackage file
    output_gpkg = "outputs/geopackages/ecoregions_with_clusters.gpkg"
    merged_gdf.to_file(output_gpkg, driver="GPKG", layer="ecoregions_clusters")
    print(f" Merged GeoPackage saved at: {output_gpkg}")

    # Step 5: Load the merged GeoPackage
    gpkg_path = "outputs/geopackages/ecoregions_with_clusters.gpkg"
    layer_name = "ecoregions_clusters_2"
    ecoregions_gdf = gpd.read_file(gpkg_path, layer=layer_name)

    # Step 6: Handle NaN Values in Cluster Column
    if "Cluster" not in ecoregions_gdf.columns:
        raise ValueError(" The 'Cluster' column is missing in the GeoPackage!")

    # Step 7: Get unique valid clusters (excluding -1)
    valid_clusters = sorted([c for c in ecoregions_gdf["Cluster"].unique() if c != -1])
    num_clusters = len(valid_clusters)

    # Step 8: Define a Custom Colormap
    # Define custom colorblind-friendly palette
    # colors = ['#0072B2', '#009E73', '#E69F00', '#CC79A7']  # CUD palette
    # colors = ['#1b9e77',  # green-teal
    #       '#7570b3',  # purple-blue
    #       '#d95f02',  # orange
    #       '#e7298a']  # pink


    colors = [
        '#332288',  # purple-blue
        '#ddcc77',  # Orange
        '#44aa99', # Yellow
        '#aa4499',  # Bluish Green
        ]  

    # colors = ['#4B0082', '#DAA520', '#008080', '#DC143C']
    cmap = ListedColormap(colors)
    # cmap = plt.get_cmap("cividis", num_clusters)  # Use a discrete colormap for clusters
    norm = mcolors.Normalize(vmin=min(valid_clusters), vmax=max(valid_clusters))

    # Step 9: Assign Colors (Make `-1` Transparent)
    color_dict = {c: cmap(norm(c)) for c in valid_clusters if c != -1}  # Normal clusters
    color_dict[-1] = (0, 0, 0, 0)  # RGBA → Transparent for `-1`

    # Step 10: Map Colors to Data
    ecoregions_gdf["color"] = ecoregions_gdf["Cluster"].map(color_dict)

    # Remove Ecoregions without Primary Forest
    ecoregions_gdf = ecoregions_gdf.dropna(subset=["Cluster"])

    # Count occurrences of each category
    counts = ecoregions_gdf["Cluster"].value_counts().sort_index()
    labels = counts.index
    dynamic_labels = [label for label in labels]

    # cmap = cm.viridis
    colors = []

    for label in labels:
        idx = dynamic_labels.index(label)
        colors.append(cmap(idx / (len(dynamic_labels) - 1)))

    # Compute total area per cluster
    # Compute total area per cluster and normalize to get proportions
    area_per_cluster = ecoregions_gdf.groupby("Cluster")["area km2"].sum()
    area_per_cluster = area_per_cluster[area_per_cluster.index.isin(labels)]  # ensure consistent ordering
    area_proportions = area_per_cluster / area_per_cluster.sum()
    print(f"Area proportion\n{area_proportions}")
    # *************************************************************************

    # *****  Step 11: Create a Figure ***** 
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    # Step 12: Plot the Map with Transparent `-1`
    ecoregions_gdf.plot(color=ecoregions_gdf["color"], edgecolor="black", linewidth=0.05, ax=ax)

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
    # *************************************************************************

    inset_ax1 = inset_axes(
        ax, width="20%", height="20%",
        loc='lower left',
        bbox_to_anchor=(0.05, 0.0, 1, 1),  # lower position than area chart
        bbox_transform=ax.transAxes,
        borderpad=2
    )

    # Normalize counts to proportions
    count_proportions = counts / counts.sum()
    bar_labels = [str(label) for label in count_proportions.index]
    bar_colors = [
        cmap(dynamic_labels.index(label) / (len(dynamic_labels) - 1))
        for label in count_proportions.index
    ]

    # Plot bar chart
    bars1 = inset_ax1.bar(bar_labels, count_proportions, color=bar_colors, edgecolor='black')

    # Add percentage labels on top of each bar
    for bar in bars1:
        height = bar.get_height()
        inset_ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f'{height * 100:.1f}%',
            ha='center', va='bottom', fontsize=7, color='black'
        )

    # Style: transparent background and subtle gray frame
    inset_ax1.set_facecolor((1, 1, 1, 0.0))  # transparent background
    for spine in inset_ax1.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)
        spine.set_alpha(0.3)

    # Remove ticks for clean look
    inset_ax1.set_xticks([])
    inset_ax1.set_yticks([])
    inset_ax1.spines['top'].set_visible(False)
    inset_ax1.spines['right'].set_visible(False)
    inset_ax1.spines['left'].set_visible(False)
    inset_ax1.spines['bottom'].set_visible(False)
    inset_ax1.set_title("Ecoregion Count (% of 150)", fontsize=9)

    # PIE 2: Area proportions (just above the first pie)
    inset_ax2 = inset_axes(
        ax,
        width="20%",
        height="20%",
        loc='lower left',
        borderpad=2,
        bbox_to_anchor=(0.05, 0.28, 1, 1),
        bbox_transform=ax.transAxes
    )
    # Plot bar chart
    bar_labels = [str(label) for label in area_proportions.index]
    bar_colors = [
        cmap(dynamic_labels.index(label) / (len(dynamic_labels) - 1))
        for label in area_proportions.index
    ]

    bars = inset_ax2.bar(bar_labels, area_proportions, color=bar_colors, edgecolor='black')

    # Formatting
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        inset_ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,  # offset above the bar
            f'{height*100:.1f}%',  # as percent
            ha='center', va='bottom', fontsize=7, color='black'
        )

    # Clean visual style: no ticks, no axis frame
    inset_ax2.set_xticks([])
    inset_ax2.set_yticks([])
    inset_ax2.spines['top'].set_visible(False)
    inset_ax2.spines['right'].set_visible(False)
    inset_ax2.spines['left'].set_visible(False)
    inset_ax2.spines['bottom'].set_visible(False)
    inset_ax2.set_facecolor((1, 1, 1, 0.0))  # fully transparent background
    # Optional title
    inset_ax2.set_title("Area Proportions (%)", fontsize=9)

    # *************************************************************************
    # Add cluster legend manually
    if "Cluster" in ecoregions_gdf.columns:
        cluster_ids = sorted(ecoregions_gdf["Cluster"].dropna().unique())
        cmap =  cmap#plt.get_cmap("cividis")
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
    plt.tight_layout()

    plt.savefig("./outputs/figures/Ecoregions_Cluster.png", dpi=300, bbox_inches='tight')

    # ✅ Step 11: Force Show Plot
    plt.show()


if __name__ == '__main__':
    run()