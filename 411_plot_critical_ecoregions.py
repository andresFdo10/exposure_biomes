import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches

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
    ax.gridlines(alpha=0.6, linewidth=0.2, draw_labels=False)
    # ax.gridlines(alpha=0.6, linewidth=0.2, draw_labels=True,
    #         #  xlabels_top=False, xlabels_bottom=True,
    #         #  ylabels_left=True, ylabels_right=False,
    #          x_inline=False, y_inline=False)

def perform_pca(dataframe, columns, variance_threshold=0.9, n_components=3, n_clusters=4):
    """
    Perform PCA on selected columns of a dataframe and apply K-Means clustering.
    """

    # Select relevant columns but retain index
    df = dataframe[columns].copy()
    
    # Drop rows with missing values while keeping track of the retained ECO_IDs
    df = df.dropna()
    # df = df.fillna(0)
    retained_indices = df.index  # Save ECO_IDs of remaining rows

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Perform PCA with the specified number of components
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df_scaled)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained variance ratio:\n", explained_variance_ratio)
    # Compute explained variance percentages
    explained_var_pc1 = explained_variance_ratio[0] * 100  # Percentage for PC1
    explained_var_pc2 = explained_variance_ratio[1] * 100  # Percentage for PC2

    # Principal component loadings (eigenvectors)
    pca_loadings = pca.components_
        # ✅ Save PCA loadings to CSV for later use
    loadings_df = pd.DataFrame(
        pca_loadings.T,  # Transpose so features are rows
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=columns  # Assign feature names to rows
    )
    loadings_df.to_csv("outputs/csv/pca_loadings.csv")  
    print("PCA loadings saved as 'outputs/csv/pca_loadings.csv' ✅")

    # Convert PCA-transformed data into a DataFrame
    PC_scores = pd.DataFrame(
        reduced_data,
        columns=[f"PC{i+1}" for i in range(n_components)]
        )
        # STEP 3. BIPLOTV
    PC1 = pca.fit_transform(df_scaled)[:,0]
    PC2 = pca.fit_transform(df_scaled)[:,1]
    
    scalePC1 = 1.0/(PC1.max() - PC1.min())
    scalePC2 = 1.0/(PC2.max() - PC2.min())

    # ✅ Apply K-Means Clustering on the PCA-transformed data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    # ✅ Add cluster labels to the DataFrame
    PC_scores["Cluster"] = clusters


    return PC_scores, explained_variance_ratio, pca_loadings, retained_indices


def plot_pca_clusters_with_map(
    PC_scores,
    pca_loadings,
    explained_variance,
    pc_x=1,
    pc_y=2,
    combined_effects_path="outputs/csv/ecoregions_combined_effects.csv",
    gpkg_path="outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg",
    layer_name="ZonalStat_Ecoregions",
    neotropic_layer=None,
    save_dir="outputs/figures"
):

    os.makedirs(save_dir, exist_ok=True)

    # Adjust for indexing
    pc_x_idx = pc_x - 1
    pc_y_idx = pc_y - 1

    PC_scores = PC_scores.dropna()
    pc_x_col = PC_scores.columns[pc_x]
    pc_y_col = PC_scores.columns[pc_y]

    # Scale factors
    scale_x = 1.0 / (PC_scores[pc_x_col].max() - PC_scores[pc_x_col].min())
    scale_y = 1.0 / (PC_scores[pc_y_col].max() - PC_scores[pc_y_col].min())

    explained_var_x = explained_variance[pc_x_idx] * 100
    explained_var_y = explained_variance[pc_y_idx] * 100

    # Load combined effects CSV
    combined_df = pd.read_csv(combined_effects_path)
    combined_df.columns = combined_df.columns.str.strip()
    if "Ecoregion ID" in combined_df.columns:
        combined_df = combined_df.rename(columns={"Ecoregion ID": "ECO_ID"})

    label = f"PC{pc_x} & PC{pc_y}"
    ids = combined_df[combined_df["Combined_Effect"] == label]["ECO_ID"].astype(float).unique()

    # Load GPKG and extract relevant geometries
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    gdf["ECO_ID"] = pd.to_numeric(gdf["ECO_ID"], errors="coerce")
    gdf = gdf.dropna(subset=["ECO_ID"])
    highlighted_map = gdf[gdf["ECO_ID"].isin(ids)]
    highlighted_scores = PC_scores[PC_scores["ECO_ID"].isin(ids)]

    # Start plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection":ccrs.PlateCarree()})

    # --- Left: PCA Biplot ---
    clusters = PC_scores.get("Cluster", None)
    scatter = ax1.scatter(
        PC_scores[pc_x_col] * scale_x,
        PC_scores[pc_y_col] * scale_y,
        c=clusters,
        cmap="viridis",
        alpha=0.7,
        s=50
    )

    # Define your cluster labels
    cluster_labels = {
        0: "Fire-Affected Forests",
        1: "ENSO-Exposed Regions",
        2: "Stable/Mixed Impact",
        3: "High Deforestation"
        # Add more if needed
    }

    PC_scores["Cluster"] = PC_scores["Cluster"].astype(int)
    # Add cluster legend manually
    if "Cluster" in PC_scores.columns:
        cluster_ids = sorted(PC_scores["Cluster"].dropna().unique())
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(cluster_ids), vmax=max(cluster_ids))

        handles = []
        for cid in cluster_ids:
            color = cmap(norm(cid))
            label = cluster_labels.get(cid, f"Cluster {cid}")
            patch = mpatches.Patch(color=color, label=label)
            handles.append(patch)


        ax1.legend(handles=handles, title="Clusters", loc="lower right", fontsize=10, title_fontsize=11)
    # clusters = PC_scores.get("Cluster", None)
    # scatter = ax1.scatter(
    #     PC_scores[pc_x_col] * scale_x,
    #     PC_scores[pc_y_col] * scale_y,
    #     c=clusters,
    #     cmap="viridis",
    #     alpha=0.7,
    #     # label="All Ecoregions",
    #     s=50
    # )

    # Highlight critical ecoregions in red
    ax1.scatter(
        highlighted_scores[pc_x_col] * scale_x,
        highlighted_scores[pc_y_col] * scale_y,
        # color='red',
        facecolor='none',
        edgecolor='red',
        # linestyle='--',
        linewidth=1.5,
        # label="Combined Effect",
        s=100,
        # zorder=3
    )

    # Annotate only red points
    # for _, row in highlighted_scores.iterrows():
    #     ax1.text(
    #         (row[pc_x_col] * scale_x) * 1.03,
    #         (row[pc_y_col] * scale_y) * 1.10,
    #         str(int(row["ECO_ID"])),
    #         fontsize=12,
    #         ha='right',
    #         color='black'
    #     )

    # Add vectors
    features = [
        'El Niño EWI',
        'La Niña EWI',
        'Forest Loss',
        'Forest Loss Rate',
        'Fire-Induced Loss %'
    ]
    for i, feature in enumerate(features):
        ax1.arrow(
            0, 0,
            pca_loadings[pc_x_idx, i],
            pca_loadings[pc_y_idx, i],
            color='black',
            head_width=0.02,
            head_length=0.02,
            alpha=0.8
        )
        ax1.text(
            pca_loadings[pc_x_idx, i] * 1.03,
            pca_loadings[pc_y_idx, i] * 1.03,
            feature,
            fontsize=12
        )

    ax1.set_xlim(-0.95, 0.95)
    ax1.set_ylim(-0.95, 0.95)
    ax1.set_aspect("equal")
    ax1.axhline(0, color="gray", linestyle="--")
    ax1.axvline(0, color="gray", linestyle="--")
    ax1.set_xlabel(f'PC{pc_x} ({explained_var_x:.2f}%)')
    ax1.set_ylabel(f'PC{pc_y} ({explained_var_y:.2f}%)')
    # ax1.set_title(f'PCA Biplot: PC{pc_x} vs PC{pc_y}')
    # ax1.legend()
    ax1.legend(handles=handles, title="Clusters", loc="lower right", fontsize=10, title_fontsize=11)

    # --- Right: Map Plot ---
    # gdf.boundary.plot(
    #     ax=ax2,
    #     color="lightgray",
    #     linewidth=0.2,
    #     alpha=0.2
    #     )
    # neotropic_layer.boundary.plot(
    #     ax=ax2,
    #     color="lightgray",
    #     linewidth=0.4,
    #     # alpha=0.7
    #     )
    extent  = (-105, -33.9, -31.5, 23.8)
    # RGBA for black with 50% transparency
    edge_color = (1, 1, 1, 0.3)  # R, G, B, Alpha
    if not highlighted_map.empty:
        highlighted_map.plot(
            ax=ax2, 
            color='red', 
            # facecolor='none',
            alpha=0.6,
            edgecolor=edge_color, 
            # linestyle='--',
            linewidth=0.2
            )
        # ax2.set_title(f"Ecoregions with {label}")
    # else:
    #     ax2.set_title(f"No ecoregions found for {label}")
    ax2.axis("off")
    add_features(ax2, extent=extent)

    # Save and show
    filename = f"{save_dir}/PCA_PC{pc_x}_vs_PC{pc_y}_with_map.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Figure saved: {filename}")
    plt.show()


def run():

    # path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"
    # layer_name = "zonal_statistics_v2"
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"
    layer_name = "ZonalStat_Ecoregions"

    # Load the GeoPackage file
    ecoregions_gdf = gpd.read_file(path_gpkg, layer=layer_name)
    print(ecoregions_gdf.dtypes)
    neotropic = ecoregions_gdf.dissolve("REALM")
    # neotropic.plot()
    # plt.show()

    # Convert to pandas DataFrame and set ECO_ID as index
    ecoregions_df = pd.DataFrame(ecoregions_gdf)

    # Drop ecoregions where PrimaryForest is 0
    ecoregions_df = ecoregions_df.drop(
        ecoregions_df[ecoregions_df["PrimaryForest50"] == 0].index
        )

    ecoregions_df.set_index("ECO_ID", inplace=True)  # Ensure ECO_ID is the index

    selected_columns = [
        "EWMnino",
        "EWMnina",
        "PrimaryForest_Loss50",
        "PrimaryLoss_Rate50",
        "proportion_fire-induced_Primaryloss"
    ]

    # Perform PCA with K-Means clustering (e.g., 3 clusters)
    pc_analysis = perform_pca(
        ecoregions_df, 
        selected_columns, 
        variance_threshold = 0.9,
        n_components=3, 
        n_clusters=4
    )
    reduced_data = pc_analysis[0]
    variance_ratios = pc_analysis[1]
    pca_loadings = pc_analysis[2]
    retained_indices = pc_analysis[3]

    # Add back ECO_IDs from retained indices
    reduced_data.insert(0, "ECO_ID", retained_indices)  # Ensures ECO_ID is the first column

    # convert to dataframe
    df_pca_loadings = pd.DataFrame(pca_loadings)

    # add column names
    df_pca_loadings.columns = selected_columns
    # add row names
    df_pca_loadings.index = ["PC1", "PC2", "PC3"]
    
    for pc_x, pc_y in [(1, 2), (1, 3), (2, 3)]:
        plot_pca_clusters_with_map(
            reduced_data,
            pca_loadings,
            variance_ratios,
            pc_x=pc_x,
            pc_y=pc_y,
            neotropic_layer=neotropic,
        )
    

if __name__ == "__main__":
    run()