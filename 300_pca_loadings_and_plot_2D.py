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


def perform_pca(
    dataframe, 
    columns, 
    variance_threshold=0.9, 
    n_components=3, 
    n_clusters=4
):
    """
    Perform Principal Component Analysis (PCA) on selected columns of a dataframe 
    and apply K-Means clustering.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe containing the features to be analyzed.
    columns : list
        List of column names to be used for PCA.
    variance_threshold : float, optional
        Variance threshold to retain principal components (default is 0.9).
    n_components : int, optional
        Number of principal components to retain (default is 3).
    n_clusters : int, optional
        Number of clusters for K-Means clustering (default is 4).

    Returns
    -------
    PC_scores : pandas.DataFrame
        Dataframe containing the principal component scores.
    explained_variance_ratio : numpy.ndarray
        Explained variance ratio for each principal component.
    pca_loadings : numpy.ndarray
        Principal component loadings (eigenvectors).
    retained_indices : pandas.Index
        Index of retained ECO_IDs.
    pca : sklearn.decomposition.PCA
        PCA object used for transformation.
    eigenvalues : numpy.ndarray
        Eigenvalues of the principal components.
    """

    # Select relevant columns but retain index
    df = dataframe[columns].copy()
    
    # Drop rows with missing values while keeping track of the retained ECO_IDs
    df = df.dropna()
    # df = df.fillna(0)
    retained_indices = df.index  # Save ECO_IDs of remaining rows

    # Compute correlation matrix
    corr_matrix = df.corr()
    # Save correlation matrix to CSV
    corr_matrix.to_csv("outputs/csv/correlation_matrix.csv")    
    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=0.5
        )
    plt.title("Correlation Matrix of Selected Variables")
    plt.show()

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Perform PCA with the specified number of components
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df_scaled)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained variance ratio:\n", explained_variance_ratio)
    eigenvalues = pca.explained_variance_
    print("Explained variance ratio:\n", eigenvalues)
    # Compute explained variance percentages
    explained_var_pc1 = explained_variance_ratio[0] * 100  # Percentage for PC1
    explained_var_pc2 = explained_variance_ratio[1] * 100  # Percentage for PC2

    # Principal component loadings (eigenvectors)
    pca_loadings = pca.components_
        # Save PCA loadings to CSV for later use
    loadings_df = pd.DataFrame(
        pca_loadings.T,  # Transpose so features are rows
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=columns  # Assign feature names to rows
    )
    loadings_df.to_csv("outputs/csv/pca_loadings.csv")  
    print("PCA loadings saved as 'outputs/csv/pca_loadings.csv' ")

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

    # Apply K-Means Clustering on the PCA-transformed data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    # Add cluster labels to the DataFrame
    PC_scores["Cluster"] = clusters


    return PC_scores, explained_variance_ratio, pca_loadings, retained_indices, pca, eigenvalues


def plot_pca_clusters(PC_scores, pca_loadings, explained_variance, pc_x=1, pc_y=2):
    """
    Plots PCA clusters for different PC combinations dynamically with scaling.

    Parameters
    ----------
    PC_scores : pandas.DataFrame
        Dataframe containing the principal component scores.
    pca_loadings : numpy.ndarray
        Principal component loadings (eigenvectors).
    explained_variance : numpy.ndarray
        Explained variance ratio for each principal component.
    pc_x : int, optional
        X-axis principal component index (default is 1).
    pc_y : int, optional
        Y-axis principal component index (default is 2).
    """
    pc_x -= 1  # Adjust index (PC1 -> index 0, PC2 -> index 1, etc.)
    pc_y -= 1

    # Drop NaN values
    PC_scores = PC_scores.dropna()

    # Adjust column indices (skip 'ECO_ID')
    pc_x += 1  
    pc_y += 1  

    # Print available columns
    print("Columns in PC_scores:", PC_scores.columns)

    # Compute explained variance percentages
    explained_var_x = explained_variance[pc_x - 1] * 100
    explained_var_y = explained_variance[pc_y - 1] * 100

    # Compute correct min/max values for scaling
    min_x, max_x = PC_scores.iloc[:, pc_x].min(), PC_scores.iloc[:, pc_x].max()
    min_y, max_y = PC_scores.iloc[:, pc_y].min(), PC_scores.iloc[:, pc_y].max()

    # Print min/max for debugging
    print(f"PC{pc_x} min/max: {min_x}, {max_x}")
    print(f"PC{pc_y} min/max: {min_y}, {max_y}")

    # Adjust scaling factors (increase factor to avoid collapsing)
    scalePC_x = 1.0 / (max_x - min_x) if max_x - min_x != 0 else 1
    scalePC_y = 1.0 / (max_y - min_y) if max_y - min_y != 0 else 1
    print(f"Scaling factors - X: {scalePC_x}, Y: {scalePC_y}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Ensure 'Cluster' exists before using it
    if 'Cluster' in PC_scores.columns:
        clusters = PC_scores['Cluster']
    else:
        print("Warning: 'Cluster' column not found! Using default.")
        clusters = None  # Avoid crashing

    # Fix scatter plot: Proper column selection, remove scaling temporarily
    scatter = ax.scatter(
        PC_scores.iloc[:, pc_x],
        PC_scores.iloc[:, pc_y], 
        c=clusters, 
        cmap='viridis', 
        alpha=0.7,
    )



    # Add legend if clusters exist
    if clusters is not None:
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)


    # Plot PCA vectors (loadings)
    features = [
        'El Niño EWI',
        'La Niña EWI',
        'Forest Loss',
        'Forest Loss Rate',
        'Fire-Induced Loss %'
        ]

    for i, feature in enumerate(features):
        ax.arrow(
            0, 
            0, 
            pca_loadings[pc_x - 1, i], 
            pca_loadings[pc_y - 1, i],
            color='black', 
            alpha=0.8, 
            head_width=0.02, 
            head_length=0.02
            )
        ax.text(
            pca_loadings[pc_x - 1, i] * 1.03,
            pca_loadings[pc_y - 1, i] * 1.03,
            feature,
            fontsize=12
            )

    # Annotate ECO_ID labels on the biplot
    for i, row in PC_scores.iterrows():
        ax.text(
            row.iloc[pc_x] * scalePC_x, 
            row.iloc[pc_y] * scalePC_y, 
            str(row["ECO_ID"]),
            fontsize=8,
            ha='right', 
            alpha=0.7, 
            color='black'
            )

    # Set axis limits and aspect ratio
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_aspect('equal', adjustable='datalim')

    # Add reference lines at (0,0)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    # Set axis labels with explained variance
    ax.set_xlabel(f'PC{pc_x} ({explained_var_x:.2f}%)', fontsize=14)
    ax.set_ylabel(f'PC{pc_y} ({explained_var_y:.2f}%)', fontsize=14)

    # Export the plot
    # Define the directory where the file will be saved
    output_dir = "./outputs/figures/"
    # Create the directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"PC{pc_x}_vs_PC{pc_y}.png"), 
        dpi=300,
        bbox_inches='tight'
        )

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

    # Save the plot
    output_dir = "./outputs/figures/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "elbow_method.png"), dpi=300)

    # Show plot
    plt.show()

def add_literals(axes, literal, x, y):
    """
    Add a text annotation to a plot with specific styling.

    Parameters:
    - axes: The matplotlib axes object where the text will be added.
    - literal: The string text to display.
    - x: The x-coordinate for the text position, in axes-relative coordinates.
    - y: The y-coordinate for the text position, in axes-relative coordinates.
    """
    # Add the text annotation to the plot
    axes.text(
        x, 
        y, 
        literal,
        transform=axes.transAxes,  # Use axes coordinates
        fontsize=12, 
        fontweight='bold', 
        va='top',  # Vertical alignment
        ha='left', # Horizontal alignment
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  # Background styling
    )
def determine_optimal_clusters(data, min_clusters=2, max_clusters=10):
    """
    Determines the optimal number of clusters using the Elbow Method and the Silhouette Score.

    Parameters:
    - data: PCA-transformed data (numpy array or DataFrame)
    - min_clusters: Minimum number of clusters to test
    - max_clusters: Maximum number of clusters to test

    Returns:
    - Optimal cluster count based on silhouette score
    """
    inertias = []
    silhouette_scores = []

    # Iterate over possible cluster counts
    for k in range(min_clusters, max_clusters + 1):
        # Compute k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        # Store inertia (sum of squared distances to cluster centers)
        inertias.append(kmeans.inertia_)

        # Compute silhouette score (only if k > 1)
        if k > 1:
            # Silhouette score measures separation and cohesion of clusters
            # Range: [-1, 1], with higher values indicating better clustering
            silhouette_scores.append(silhouette_score(data, cluster_labels))
        else:
            # Silhouette score is undefined for k=1
            silhouette_scores.append(np.nan)

    # Plot Elbow Method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(range(min_clusters, max_clusters + 1), inertias, marker='o', linestyle='--')
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster Variance)")
    ax1.set_title("Elbow Method for Optimal k")
    ax1.grid(True)
    literal = "(a)"
    add_literals(ax1, literal, 0.85, 0.94)

    # Plot Silhouette Score
    ax2.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', linestyle='--', color='red')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score for Optimal k")
    ax2.grid(True)
    literal = "(b)"
    add_literals(ax2, literal, 0.85, 0.94)

    # Show plots
    plt.tight_layout()
    plt.savefig("outputs/figures/elbow_method_and_silhouette_score.png", dpi=300)
    plt.show()

    # Determine optimal k from silhouette score
    optimal_k = range(min_clusters, max_clusters + 1)[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

    return optimal_k

def plot_explained_variance(pca_model):
    """
    Plots cumulative explained variance to help select the number of components.
    """
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=70, color='r', linestyle='--', label='70% Threshold')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('Explained Variance by Number of Components')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/pca_explained_variance.png", dpi=300)
    plt.show()

def plot_scree_plot(pca_model):
    """
    Plots the scree plot to help select the number of components.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca_model)+1), pca_model, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Explained Variance')
    plt.title('Scree Plot')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("outputs/figures/pca_scree_plot.png", dpi=300)
    plt.show()

def plot_scree_plot_kr(pca_model):
    """
    Plots the scree plot to help select the number of components.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca_model)+1), pca_model, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Explained Variance')
    plt.title('Scree Plot')
    plt.axhline(y=1, color='r', 
            linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("outputs/figures/pca_scree_plot.png", dpi=300)
    plt.show()


def run():
    """
    Run the PCA and clustering analysis on ecoregion data.

    This function loads a GeoPackage file, performs PCA on selected columns,
    applies K-Means clustering, and generates various plots to visualize
    the PCA and clustering results.

    Parameters:
    None

    Returns:
    None
    """
    # Define path and layer for GeoPackage
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
    ecoregions_df.set_index("ECO_ID", inplace=True)

    # Columns to include in PCA
    selected_columns = [
        "EWMnino",
        "EWMnina",
        "PrimaryForest_Loss50",
        "PrimaryLoss_Rate50",
        "proportion_fire-induced_Primaryloss"
    ]

    # Perform PCA and K-Means clustering
    pc_analysis = perform_pca(
        ecoregions_df,
        selected_columns,
        variance_threshold=0.9,
        n_components=5,
        n_clusters=4
    )
    reduced_data = pc_analysis[0]  # Reduced data after PCA
    variance_ratios = pc_analysis[1]  # Explained variance ratios
    pca_loadings = pc_analysis[2]  # Loadings for each component
    retained_indices = pc_analysis[3]  # Indices of retained rows
    pca_model = pc_analysis[4]  # PCA model
    eigenvalues = pc_analysis[5]  # Eigenvalues of components

    print(f"Ratios: \n{variance_ratios}")

    # Add back ECO_IDs for the reduced data
    reduced_data.insert(0, "ECO_ID", retained_indices)

    # Save PCA-transformed data with clusters
    reduced_data.to_csv(
        "outputs/csv/pca_reduced_data_with_clusters.csv",
        index=False
    )
    print("PCA-transformed data with Clusters saved as 'outputs/csv/pca_reduced_data_with_clusters.csv' ✅")

    # Convert PCA loadings to DataFrame
    df_pca_loadings = pd.DataFrame(pca_loadings)
    df_pca_loadings.columns = selected_columns  # Assign column names
    df_pca_loadings.index = ["PC1", "PC2", "PC3", "PC4", "PC5"]  # Assign row names

    print("pca_loadings:\n", df_pca_loadings.T)

    # Save PCA loadings to CSV
    df_pca_loadings.T.to_csv(
        "outputs/csv/pca_loadings.csv",
        index=True
    )
    print("PCA Loadings saved as 'outputs/csv/pca_loadings.csv' ✅")

    # Plot elbow method to determine optimal clusters
    plot_elbow_method(reduced_data)

    # Plot PCA clusters
    plot_pca_clusters(
        reduced_data,
        pca_loadings,
        variance_ratios,
        pc_x=1,
        pc_y=2,
    )

    # Determine optimal number of clusters
    optimal_k = determine_optimal_clusters(reduced_data, min_clusters=2, max_clusters=10)

    # Plot explained variance
    plot_explained_variance(pca_model)

    # Plot scree plot
    plot_scree_plot(variance_ratios)

    # Plot scree plot with Kaiser rule
    plot_scree_plot_kr(eigenvalues)


if __name__ == "__main__":
    run()

