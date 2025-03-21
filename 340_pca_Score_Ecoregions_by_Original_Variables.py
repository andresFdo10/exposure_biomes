import pandas as pd

def run():
    # Load PCA-transformed data (ECO_ID, PC1, PC2, PC3)
    pca_df = pd.read_csv("outputs/csv/pca_reduced_data_with_clusters.csv")

    # Load PCA loadings and rename the first column (assumed to hold variable names)
    loadings_df = pd.read_csv("outputs/csv/pca_loadings.csv")
    loadings_df.rename(columns={loadings_df.columns[0]: "Variable"}, inplace=True)
    loadings_df.set_index("Variable", inplace=True)  # Set variable names as index

    # Keep only PC scores
    pc_scores = pca_df[["ECO_ID", "PC1", "PC2", "PC3"]].copy()

    # Create a new DataFrame for variable-specific influence scores
    influence_scores = pd.DataFrame()
    influence_scores["ECO_ID"] = pc_scores["ECO_ID"]

    # Compute influence scores for each variable using matrix multiplication
    for variable in loadings_df.index:
        weights = loadings_df.loc[variable].values  # Extract loadings for this variable
        influence = pc_scores[["PC1", "PC2", "PC3"]].values @ weights  # Compute weighted influence
        influence_scores[variable] = influence  # Store result

    # Save results
    influence_scores.to_csv("outputs/csv/ecoregion_variable_influence_scores.csv", index=False)

    # Preview output
    print("Ecoregion variable influence scores saved as 'outputs/csv/ecoregion_variable_influence_scores.csv'")
    print(influence_scores.head())
if __name__ == "__main__":
    run()