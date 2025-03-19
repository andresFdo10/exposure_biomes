import pandas as pd
import numpy as np
import plotly.express as px
import ace_tools_open as tools

# Load PCA-transformed data and loadings
pca_df = pd.read_csv("outputs/csv/pca_reduced_data_with_clusters.csv")
loadings_df = pd.read_csv("outputs/csv/pca_loadings.csv", index_col=0)  # Load loadings

# Display loadings for reference
print("PCA Loadings:\n", loadings_df)

# Now you can use `loadings_df` to apply correct filtering logic


# Extract the relevant loadings (assuming columns correspond to variables)
loadings_PC1 = loadings_df.iloc[:, 0]  # First PC
loadings_PC2 = loadings_df.iloc[:, 1]  # Second PC
loadings_PC3 = loadings_df.iloc[:, 2]  # Third PC

# **Determine Filtering Conditions Based on Loadings**
# Fire-Induced Loss: PC1 (-), PC2 (-), PC3 (+)
# ENSO Effects: PC1 (+), PC2 (+)
# Forest Loss Rate: Multiple PCs (depends on the values)

# Set 75th percentile thresholds for each PC score
threshold_PC1 = np.percentile(abs(pca_df["PC1"]), 1)
threshold_PC2 = np.percentile(abs(pca_df["PC2"]), 1)
threshold_PC3 = np.percentile(abs(pca_df["PC3"]), 1)

loading_threshold = 0.4  # Adjusted for less strict filtering


filtered_df = pca_df[
    (abs(pca_df["PC1"]) >= threshold_PC1) &  
    (abs(pca_df["PC2"]) >= threshold_PC2) &  
    (abs(pca_df["PC3"]) >= threshold_PC3) &  
    (abs(loadings_df.loc["EWMnino", "PC1"]) > loading_threshold) &  
    (abs(loadings_df.loc["EWMnina", "PC2"]) > loading_threshold) &  
    (abs(loadings_df.loc["PrimaryLoss_Fires50%", "PC3"]) > loading_threshold) &  
    (abs(loadings_df.loc["primaryLoss_rate", "PC2"]) > loading_threshold)
]

# Display the filtered dataframe
tools.display_dataframe_to_user(name="Filtered Ecoregions with All Effects", dataframe=filtered_df)

# **Create a 3D scatter plot highlighting these points**
fig = px.scatter_3d(
    pca_df, 
    x="PC1", 
    y="PC2", 
    z="PC3", 
    color="Cluster",
    opacity=0.5,
    title="Ecoregions Strongly Affected by Climate Variability, Deforestation, and Fire",
    hover_data={"ECO_ID": True}
)

# Highlight the filtered points in red
fig.add_trace(px.scatter_3d(
    filtered_df, 
    x="PC1", 
    y="PC2", 
    z="PC3", 
    color_discrete_sequence=["red"], 
    opacity=1.0, 
    hover_data={"ECO_ID": True}
).data[0])

# Show plot
fig.show()
