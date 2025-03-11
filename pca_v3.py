import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_pca(dataframe, columns, variance_threshold=0.9, n_components=3):
    """
    Perform PCA on selected columns of a dataframe and reduce dimensions to the desired number of components.
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
    reduced_data = pd.DataFrame(pca.fit_transform(df_scaled), columns=["PC1", "PC2", "PC3"] )

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Principal component loadings (eigenvectors)
    pca_loadings = pca.components_

    return reduced_data, explained_variance_ratio, pca_loadings, retained_indices


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
    print("PCA-transformed data saved as 'outputs/csv/pca_reduced_data.csv' ✅")

if __name__ == "__main__":
    run()
