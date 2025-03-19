import pandas as pd
import numpy as np
import plotly.express as px
# Display the filtered dataframe
import ace_tools_open as tools

# Load PCA-transformed data
pca_df = pd.read_csv("outputs/csv/pca_reduced_data_with_clusters.csv")

# Define threshold (e.g., 75th percentile for strong influence)
threshold_PC1 = np.percentile(abs(pca_df["PC1"]), 75)
threshold_PC2 = np.percentile(abs(pca_df["PC2"]), 75)
threshold_PC3 = np.percentile(abs(pca_df["PC3"]), 75)

# Filter points where all three PCs exceed their thresholds
filtered_df = pca_df[
    (abs(pca_df["PC1"]) >= threshold_PC1) &
    (abs(pca_df["PC2"]) >= threshold_PC2) &
    (abs(pca_df["PC3"]) >= threshold_PC3)
]

tools.display_dataframe_to_user(name="Filtered Ecoregions with All Effects", dataframe=filtered_df)

# Create a 3D scatter plot highlighting these points
fig = px.scatter_3d(
    pca_df, 
    x="PC1", 
    y="PC2", 
    z="PC3", 
    color="Cluster",
    opacity=0.5,
    title="Ecoregions Strongly Affected by All Three Effects",
    hover_data={"ECO_ID": True}  # Show ECO_ID when hovering
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