import plotly.express as px
import pandas as pd

def run():

    # Load PCA-transformed data
    pca_df = pd.read_csv("outputs/csv/pca_reduced_data.csv")

    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        pca_df[["PC1", "PC2", "PC3", "Cluster"]], 
        x="PC1", 
        y="PC2", 
        z="PC3", 
        color="Cluster",  # Color by cluster
        title="Interactive 3D PCA Scatter Plot",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "PC3": "Principal Component 3"},
        opacity=0.8
    )

    # Show interactive plot
    fig.show()

if __name__ == "__main__":
    run()