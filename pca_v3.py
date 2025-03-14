import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


import matplotlib.pyplot as plt

def plot_pca_clusters(PC_scores, pca_loadings, explained_variance, pc_x=1, pc_y=2):
    """
    Plots PCA clusters for different PC combinations dynamically with scaling.
    """
    pc_x -= 1  # Adjust index (PC1 -> index 0, PC2 -> index 1, etc.)
    pc_y -= 1

    # ✅ Drop NaN values
    PC_scores = PC_scores.dropna()

    # ✅ Adjust column indices (skip 'ECO_ID')
    pc_x += 1  
    pc_y += 1  

    # ✅ Print available columns
    print("Columns in PC_scores:", PC_scores.columns)

    # Compute explained variance percentages
    explained_var_x = explained_variance[pc_x - 1] * 100
    explained_var_y = explained_variance[pc_y - 1] * 100

    # ✅ Compute correct min/max values for scaling
    min_x, max_x = PC_scores.iloc[:, pc_x].min(), PC_scores.iloc[:, pc_x].max()
    min_y, max_y = PC_scores.iloc[:, pc_y].min(), PC_scores.iloc[:, pc_y].max()

    # ✅ Print min/max for debugging
    print(f"PC{pc_x} min/max: {min_x}, {max_x}")
    print(f"PC{pc_y} min/max: {min_y}, {max_y}")

    # ✅ Adjust scaling factors (increase factor to avoid collapsing)
    scalePC_x = 1.0 / (max_x - min_x) if max_x - min_x != 0 else 1
    scalePC_y = 1.0 / (max_y - min_y) if max_y - min_y != 0 else 1
    print(f"Scaling factors - X: {scalePC_x}, Y: {scalePC_y}")

    # ✅ Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # ✅ Ensure 'Cluster' exists before using it
    if 'Cluster' in PC_scores.columns:
        clusters = PC_scores['Cluster']
    else:
        print("Warning: 'Cluster' column not found! Using default.")
        clusters = None  # Avoid crashing

    # ✅ Fix scatter plot: Proper column selection, remove scaling temporarily
    scatter = ax.scatter(
        PC_scores.iloc[:, pc_x] * scalePC_x,  # Scale by 10 to fix visibility
        PC_scores.iloc[:, pc_y] * scalePC_y, 
        c=clusters, cmap='viridis', alpha=0.7
    )

    # ✅ Add legend if clusters exist
    if clusters is not None:
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

    # ✅ Plot PCA vectors (loadings)
    features = ['El Niño EWI', 'La Niña EWI', 'Forest Loss Rate', 'Fire-Induced Loss %']
    for i, feature in enumerate(features):
        ax.arrow(0, 0, pca_loadings[pc_x - 1, i], pca_loadings[pc_y - 1, i], 
                 color='black', alpha=0.8, head_width=0.02, head_length=0.02)
        ax.text(pca_loadings[pc_x - 1, i] * 1.03, pca_loadings[pc_y - 1, i] * 1.03, 
                feature, fontsize=12)

    # ✅ Set axis limits and aspect ratio
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_aspect('equal', adjustable='datalim')

    # ✅ Add reference lines at (0,0)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    # ✅ Set axis labels with explained variance
    ax.set_xlabel(f'PC{pc_x} ({explained_var_x:.2f}%)', fontsize=14)
    ax.set_ylabel(f'PC{pc_y} ({explained_var_y:.2f}%)', fontsize=14)
    plt.savefig(f"./outputs/figures/PC{pc_x}_vs_PC{pc_y}.png", dpi=300, bbox_inches='tight')


    # Show plot
    plt.show()



def plot_elbow_method(data, max_clusters=10):
    """
    Plots the Elbow Method to determine the optimal number of clusters.
    
    Parameters:
    - data: PCA-transformed data (numpy array or DataFrame)
    - max_clusters: Maximum number of clusters to test (default: 10)
    """
    inertias = []

    # Compute K-Means clustering for different cluster sizes
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)  # Store inertia (within-cluster variance)

    # Plot the Elbow Graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-Cluster Variance)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)

    # Show plot
    plt.show()

def find_best_clusters_silhouette(data, min_clusters=2, max_clusters=10):
    """
    Computes silhouette scores for different cluster numbers to determine the optimal k.

    Parameters:
    - data: PCA-transformed data
    - min_clusters: Minimum number of clusters to test
    - max_clusters: Maximum number of clusters to test
    """
    silhouette_scores = []

    # Compute silhouette score for different cluster sizes
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)

    # Plot the Silhouette Score Graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', linestyle='--', color='red')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal k")
    plt.grid(True)

    # Show plot
    plt.show()


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

def run():
    # Example usage
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg"
    layer_name = "zonal_statistics_v2"

    # Load the GeoPackage file
    ecoregions_gdf = gpd.read_file(path_gpkg, layer=layer_name)

    # Convert to pandas DataFrame and set ECO_ID as index
    ecoregions_df = pd.DataFrame(ecoregions_gdf)
    ecoregions_df.set_index("ECO_ID", inplace=True)  # Ensure ECO_ID is the index

    # Select relevant columns
    selected_columns = [
        "EWMnino",
        "EWMnina",
        "primaryLoss_rate",
        "PrimaryLoss_Fires50%"
    ]

    # Perform PCA with K-Means clustering (e.g., 3 clusters)
    reduced_data, variance_ratios, pca_loadings, retained_indices = perform_pca(
        ecoregions_df, selected_columns, variance_threshold=0.9, n_components=3, n_clusters=4
    )

    # Add back ECO_IDs from retained indices
    reduced_data.insert(0, "ECO_ID", retained_indices)  # Ensures ECO_ID is the first column

    # Save the PCA-transformed data with Clusters and ECO_ID
    reduced_data.to_csv("outputs/csv/pca_reduced_data_with_clusters.csv", index=False)
    print("PCA-transformed data with Clusters saved as 'outputs/csv/pca_reduced_data_with_clusters.csv' ✅")
    print(reduced_data)
    
    plot_elbow_method(reduced_data)  # Use PCA-reduced data
    find_best_clusters_silhouette(reduced_data)
    plot_pca_clusters(reduced_data, pca_loadings, variance_ratios, pc_x=1, pc_y=3)


if __name__ == "__main__":
    run()

