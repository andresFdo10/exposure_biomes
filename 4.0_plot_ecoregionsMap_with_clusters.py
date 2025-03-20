import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def run():

    # ✅ Step 1: Load the original GeoPackage with ecoregions
    gpkg_path = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"  # Update this if needed
    layer_name = "zonal_statistics_v2"  # Update if your layer name is different
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

    # ✅ Step 7: Create a Figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # ✅ Step 8: Plot the Map with Transparent `-1`
    ecoregions_gdf.plot(
        color=ecoregions_gdf["color"],  # Uses pre-defined colors
        edgecolor="black", 
        linewidth=0.05,
        ax=ax
    )

    # ✅ Step 9: Add a Color Legend (Exclude -1)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_dict[c], markersize=10) 
            for c in valid_clusters if c != -1]
    labels = [f"Cluster {int(c)}" for c in valid_clusters if c != -1]
    ax.legend(handles, labels, title="Clusters", loc="upper right")

    # ✅ Step 10: Finalize Plot
    ax.set_title("Ecoregions Colored by Clusters (Transparent -1)", fontsize=16)
    ax.set_axis_off()  # Hide axis for a cleaner look

    plt.savefig("./outputs/figures/Ecoregions_Cluster.png", dpi=300, bbox_inches='tight')

    # ✅ Step 11: Force Show Plot
    plt.show()


if __name__ == '__main__':
    run()