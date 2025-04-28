import pandas as pd
import numpy as np

def run():
    """
    Identify ecoregions strongly influenced by principal components and their combinations.

    This function loads PCA-transformed data and loadings, identifies ecoregions
    significantly influenced by each principal component (PC), and determines 
    ecoregions affected by combinations of PCs. The results are saved to a CSV file.

    Returns:
    None
    """

    # Load PCA-transformed dataset with ecoregion IDs and principal components
    pca_df = pd.read_csv("outputs/csv/pca_reduced_data_with_clusters.csv")

    # Load PCA loadings to understand variable influence on each PC
    loadings_df = pd.read_csv("outputs/csv/pca_loadings.csv", index_col=0)

    # Define thresholds for identifying significant effects (top and bottom 10% of PC values)
    threshold_high = 0.75  # Top 10%
    threshold_low = 0.10   # Bottom 10%

    # Identify ecoregions strongly influenced by each PC
    top_pc1 = pca_df[pca_df["PC1"] >= pca_df["PC1"].quantile(threshold_high)]
    bottom_pc1 = pca_df[pca_df["PC1"] <= pca_df["PC1"].quantile(threshold_low)]

    top_pc2 = pca_df[pca_df["PC2"] >= pca_df["PC2"].quantile(threshold_high)]
    bottom_pc2 = pca_df[pca_df["PC2"] <= pca_df["PC2"].quantile(threshold_low)]

    top_pc3 = pca_df[pca_df["PC3"] >= pca_df["PC3"].quantile(threshold_high)]
    bottom_pc3 = pca_df[pca_df["PC3"] <= pca_df["PC3"].quantile(threshold_low)]

    # Identify ecoregions affected by combined influences
    combined_pc1_pc2 = pd.merge(top_pc1, top_pc2, on="ECO_ID", how="inner")
    combined_pc1_pc3 = pd.merge(top_pc1, top_pc3, on="ECO_ID", how="inner")
    combined_pc2_pc3 = pd.merge(top_pc2, top_pc3, on="ECO_ID", how="inner")

    # Identify ecoregions affected by all three principal components
    combined_all = pd.merge(combined_pc1_pc2, top_pc3, on="ECO_ID", how="inner")

    # Create a summary DataFrame listing affected ecoregions
    combined_effects_df = pd.DataFrame({
        "ECO_ID": np.concatenate([
            combined_pc1_pc2["ECO_ID"].values, 
            combined_pc1_pc3["ECO_ID"].values, 
            combined_pc2_pc3["ECO_ID"].values,
            combined_all["ECO_ID"].values
        ]),
        "Combined_Effect": np.concatenate([
            ["PC1 & PC2"] * len(combined_pc1_pc2),
            ["PC1 & PC3"] * len(combined_pc1_pc3),
            ["PC2 & PC3"] * len(combined_pc2_pc3),
            ["PC1, PC2 & PC3"] * len(combined_all)
        ])
    }).drop_duplicates()

    # Save the results
    combined_effects_df.to_csv("outputs/csv/ecoregions_combined_effects.csv", index=False)

    # Display output
    print("Ecoregions with combined effects identified and saved as 'outputs/csv/ecoregions_combined_effects.csv'")
    print(combined_effects_df)

if __name__ == "__main__":
    run()