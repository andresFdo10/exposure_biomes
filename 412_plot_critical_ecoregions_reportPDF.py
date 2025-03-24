import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def perform_pca(
    data: pd.DataFrame,
    columns: list,
    explained_variance_threshold: float = 0.9,
    n_components: int = 3,
    n_clusters: int = 4
) -> (pd.DataFrame, np.ndarray, np.ndarray, pd.Index):
    """
    Perform Principal Component Analysis (PCA) on selected columns of a dataframe 
    and apply K-Means clustering.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    columns : list
        List of columns to select from the dataframe
    explained_variance_threshold : float, optional
        Threshold for the total variance explained by the principal components,
        by default 0.9
    n_components : int, optional
        Number of principal components to retain, by default 3
    n_clusters : int, optional
        Number of clusters to group the data into, by default 4

    Returns
    -------
    pca_scores : pd.DataFrame
        DataFrame with the PCA-transformed data and cluster labels
    explained_variance_ratio : np.ndarray
        Array with the explained variance ratio of each principal component
    pca_loadings : np.ndarray
        Array with the principal component loadings (eigenvectors)
    retained_indices : pd.Index
        Index of the retained ECO_IDs after dropping rows with missing values
    """

    # Select relevant columns but retain index
    selected_data = data[columns].copy()
    
    # Drop rows with missing values while keeping track of the retained ECO_IDs
    selected_data.dropna(inplace=True)
    retained_indices = selected_data.index  # Save ECO_IDs of remaining rows

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)

    # Perform PCA with the specified number of components
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Principal component loadings (eigenvectors)
    pca_loadings = pca.components_
    loadings_df = pd.DataFrame(
        pca_loadings.T,  # Transpose so features are rows
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=columns  # Assign feature names to rows
    )
    loadings_df.to_csv("outputs/csv/pca_loadings.csv", index=False)

    # Convert PCA-transformed data into a DataFrame
    pca_scores = pd.DataFrame(
        reduced_data,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Apply K-Means Clustering on the PCA-transformed data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    # Add cluster labels to the DataFrame
    pca_scores["Cluster"] = clusters

    return pca_scores, explained_variance_ratio, pca_loadings, retained_indices


def plot_pca_clusters_with_map(
    PC_scores,
    pca_loadings,
    explained_variance,
    pc_x=1,
    pc_y=2,
    combined_effects_path="outputs/csv/ecoregions_combined_effects.csv",
    gpkg_path="outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg",
    layer_name="ZonalStat_Ecoregions",
    save_dir="outputs/figures",
    pdf_writer=None
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Left: PCA Biplot ---
    clusters = PC_scores.get("Cluster", None)
    scatter = ax1.scatter(
        PC_scores[pc_x_col] * scale_x,
        PC_scores[pc_y_col] * scale_y,
        c=clusters,
        cmap="viridis",
        alpha=0.4,
        label="All Ecoregions"
    )

    # Highlight critical ecoregions in red
    ax1.scatter(
        highlighted_scores[pc_x_col] * scale_x,
        highlighted_scores[pc_y_col] * scale_y,
        color='red',
        edgecolor='black',
        label="Combined Effect",
        zorder=3
    )

    # Annotate only red points
    for _, row in highlighted_scores.iterrows():
        ax1.text(
            row[pc_x_col] * scale_x,
            row[pc_y_col] * scale_y,
            str(int(row["ECO_ID"])),
            fontsize=9,
            ha='right',
            color='red'
        )

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
    ax1.set_title(f'PCA Biplot: PC{pc_x} vs PC{pc_y}')
    ax1.legend()

    # --- Right: Map Plot ---
    if not highlighted_map.empty:
        highlighted_map.plot(ax=ax2, color='red', edgecolor='black', linewidth=0.05)
        ax2.set_title(f"Ecoregions with {label}")
    else:
        ax2.set_title(f"No ecoregions found for {label}")
    gdf.boundary.plot(ax=ax2, color="gray", linewidth=0.05, alpha=0.5)
    ax2.axis("off")

    # Save and show
    filename = f"{save_dir}/PCA_PC{pc_x}_vs_PC{pc_y}_with_map.png"
    plt.tight_layout()

    if pdf_writer:
        pdf_writer.savefig(fig)
        print(f"📝 Added PC{pc_x} vs PC{pc_y} figure to PDF")
    else:
        plt.savefig(filename, dpi=300)
        print(f"✅ Figure saved: {filename}")
        plt.show()

def run():
    """
    Main entry point for generating the PCA report PDF.

    This script will generate a PDF report with three figures:
    1. PC1 vs PC2
    2. PC1 vs PC3
    3. PC2 vs PC3

    Each figure will contain a PCA biplot with the
    ecoregions from the GeoPackage file plotted on it.
    """

    # path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"
    # layer_name = "zonal_statistics_v2"
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"
    layer_name = "ZonalStat_Ecoregions"

    # Load the GeoPackage file
    ecoregions_gdf = gpd.read_file(path_gpkg, layer=layer_name)

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
        n_components=3, 
        n_clusters=4
    )
    reduced_data = pc_analysis[0]
    variance_ratios = pc_analysis[1]
    pca_loadings = pc_analysis[2]
    retained_indices = pc_analysis[3]

    # Add back ECO_IDs from retained indices
    reduced_data.insert(0, "ECO_ID", retained_indices)  # Ensures ECO_ID is the first column

    # Save the PCA Loadings
    # convert to dataframe
    df_pca_loadings = pd.DataFrame(pca_loadings)

    # add column names
    df_pca_loadings.columns = selected_columns
    # add row names
    df_pca_loadings.index = ["PC1", "PC2", "PC3"]

    pdf_path = "outputs/figures/Combined_PCA_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        for pc_x, pc_y in [(1, 2), (1, 3), (2, 3)]:
            plot_pca_clusters_with_map(
                PC_scores=reduced_data,
                pca_loadings=pca_loadings,
                explained_variance=variance_ratios,
                pc_x=pc_x,
                pc_y=pc_y,
                pdf_writer=pdf  # 👈 pass the PDF object
            )

    print(f"📄 PDF report saved: {pdf_path}")


if __name__ == "__main__":
    run()
