import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_pca(dataframe, columns, variance_threshold=0.9, n_components=3):
    """
    Perform Principal Component Analysis (PCA) on the specified columns of a dataframe and reduce the dimensions to the desired number of components.

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

    Returns
    -------
    reduced_data : numpy.ndarray
        Reduced data after PCA.
    explained_variance_ratio : numpy.ndarray
        Explained variance ratio for each principal component.
    pca_loadings : numpy.ndarray
        Principal component loadings (eigenvectors).
    retained_indices : pandas.Index
        Index of retained ECO_IDs.
    """
    # Select relevant columns but retain index
    df = dataframe[columns].copy()
    
    # Drop rows with missing values while keeping track of the retained ECO_IDs
    df = df.dropna()
    retained_indices = df.index  # Save ECO_IDs of remaining rows

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Perform PCA with the specified number of components
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df_scaled)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Principal component loadings (eigenvectors)
    pca_loadings = pca.components_

    return reduced_data, explained_variance_ratio, pca_loadings, retained_indices


def run():
    """
    Example usage of the PCA function for dimensionality reduction.

    1. Load the GeoPackage file and convert it to a pandas DataFrame.
    2. Perform PCA on the selected columns with 3 components.
    3. Save the transformed data as a new CSV file.
    """
    # Example usage
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"
    layer_name = "ZonalStat_Ecoregions"

    # Load the GeoPackage file
    ecoregions_gdf = gpd.read_file(path_gpkg, layer=layer_name)

    # Convert to pandas DataFrame and set ECO_ID as index
    ecoregions_df = pd.DataFrame(ecoregions_gdf)
    ecoregions_df.set_index("ECO_ID", inplace=True)  # Ensure ECO_ID is the index

    # Select relevant columns
    selected_columns = [
        "EWMnino",
        "EWMnina",
        "PrimaryForest_Loss50",
        "PrimaryLoss_Rate50",
        "proportion_fire-induced_Primaryloss"
    ]

    # Perform PCA with only 3 components and keep retained indices
    reduced_data, variance_ratios, pca_loadings, retained_indices = perform_pca(
        ecoregions_df, selected_columns, variance_threshold=0.9, n_components=3
    )

    # Convert transformed data into a DataFrame
    pca_df = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(reduced_data.shape[1])])

    # Add back ECO_IDs from retained indices
    pca_df.insert(0, "ECO_ID", retained_indices)  # Ensures ECO_ID is the first column

    # Save the PCA-transformed data with ECO_ID
    pca_df.to_csv("outputs/csv/pca_reduced_data.csv", index=False)
    print("PCA-transformed data saved as 'outputs/csv/pca_reduced_data.csv' ")

if __name__ == "__main__":
    run()
