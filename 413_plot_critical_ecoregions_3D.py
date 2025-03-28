import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.image as mpimg
import numpy as np

def plot_3d_pca_with_map_side_by_side(
    pca_csv,
    combined_effects_csv,
    gpkg_path,
    layer_name,
    combined_label="PC1, PC2 & PC3",
    save_path="outputs/figures/PCA_3D_and_Map_PC1_PC2_PC3.png"
):
    # Load PCA data
    pca_df = pd.read_csv(pca_csv)
    combined_df = pd.read_csv(combined_effects_csv)
    combined_df.columns = combined_df.columns.str.strip()

    if "Ecoregion ID" in combined_df.columns:
        combined_df.rename(columns={"Ecoregion ID": "ECO_ID"}, inplace=True)

    target_ids = combined_df[combined_df["Combined_Effect"] == combined_label]["ECO_ID"].astype(float).unique()
    highlighted_df = pca_df[pca_df["ECO_ID"].isin(target_ids)]

    # --- Load GeoData for Mapping ---
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    gdf["ECO_ID"] = pd.to_numeric(gdf["ECO_ID"], errors="coerce")
    gdf = gdf.dropna(subset=["ECO_ID"])
    highlighted_map = gdf[gdf["ECO_ID"].isin(target_ids)]

    # --- Create the map plot as an image ---
    fig_map, ax = plt.subplots(figsize=(6, 6))
    highlighted_map.plot(ax=ax, color='red', edgecolor='black', linewidth=0.1)
    gdf.boundary.plot(ax=ax, color='lightgray', linewidth=0.2)
    ax.set_title("Ecoregions with Combined Effect: PC1, PC2 & PC3")
    ax.axis("off")

    # Convert Matplotlib map to image
# Convert Matplotlib map to image
    canvas = FigureCanvas(fig_map)
    canvas.draw()
    map_image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    map_image = map_image.reshape(fig_map.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig_map)


    # --- Create the 3D PCA plot with Plotly ---
    fig_3d = go.Figure()

    # All points
    fig_3d.add_trace(go.Scatter3d(
        x=pca_df["PC1"], y=pca_df["PC2"], z=pca_df["PC3"],
        mode='markers',
        marker=dict(size=6, color=pca_df["Cluster"], opacity=0.3, colorscale='Viridis'),
        text=pca_df["ECO_ID"].astype(str),
        name="All Ecoregions"
    ))

    # Highlighted red points
    fig_3d.add_trace(go.Scatter3d(
        x=highlighted_df["PC1"], y=highlighted_df["PC2"], z=highlighted_df["PC3"],
        mode='markers+text',
        marker=dict(size=7, color='red', opacity=1),
        text=highlighted_df["ECO_ID"].astype(str),
        name="Combined Effect",
        textposition="top center"
    ))

    fig_3d.update_layout(
        title="3D PCA - Ecoregions with Combined Effect",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ),
        width=1500,
        height=1000
    )

    # --- Show side-by-side layout ---

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot placeholder image for Plotly (you can just display the interactive version separately)
    ax1.text(0.5, 0.5, '3D PCA interactive plot saved separately.\nSee HTML output.',
             ha='center', va='center', fontsize=14)
    ax1.axis("off")

    # Show static map
    ax2.imshow(map_image)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Combined plot saved as: {save_path}")

    # Save interactive 3D plot separately
    fig_3d.write_html(save_path.replace(".png", ".html"))
    print(f"🧭 Interactive 3D PCA saved as: {save_path.replace('.png', '.html')}")

def run():
    plot_3d_pca_with_map_side_by_side(
    pca_csv="outputs/csv/pca_reduced_data_with_clusters.csv",
    combined_effects_csv="outputs/csv/ecoregions_combined_effects.csv",
    gpkg_path="outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg",
    layer_name="ZonalStat_Ecoregions"
)

if __name__ == "__main__":
    run()