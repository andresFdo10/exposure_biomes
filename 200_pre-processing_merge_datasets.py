import pandas as pd
import geopandas as gpd
import fiona
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape
import json


def plot_driver(geodataframe, variavle_name):
    # Visualize the deforestation rate for each ecoregion

    # plot using a grayscale colormap
    fig, ax = plt.subplots(figsize=(12, 8))

    geodataframe.plot(
        column=variavle_name,
        cmap='gray_r',
        edgecolor='black',
        linewidth=0.01,
        ax=ax,
        legend=True 
    )
    ax.grid(alpha=0.5)

    ax.set_title(f'{variavle_name}', fontsize=13)
    plt.show()

def run():
    # Read the geopackage file with the Entropy Weights Method for El Nino and La Nina
    path_gpkg = 'outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg'
    if not os.path.exists(path_gpkg):
        raise FileNotFoundError(f"File not found: {path_gpkg}")

    # Read the geopackage using geopandas
    ecoregions_exp = gpd.read_file(path_gpkg, layer="ZonalStat_Ecoregions_EWM_nino_nina")
    ecoregions_exp = ecoregions_exp.rename(columns={
        'EWMnino_median': 'EWMnino',
        'EWMnina_median': 'EWMnina'
        })
    # print(ecoregions_exp.dtypes)

    # Read the full CSV as a dataframe
    gfc_df = pd.read_csv('inputs/04_csv/ZonalStatistics_Ecoregions_Complete_v2.csv')
    print(f"\ngfc_df shape: {gfc_df.dtypes}\n")

    # rename the geometry column
    gfc_df = gfc_df.rename(columns={'.geo': 'geometry'})

    # Identify the common column
    common_column = 'ECO_NAME'

    # Merge the DataFrames based on the common column
    # Select only the two additional columns from CSV
    additional_columns = [col for col in ecoregions_exp.columns if col not in gfc_df.columns and col != common_column]
    # Merge the DataFrame with the GeoDataFrame
    gfc_df = gfc_df.merge(ecoregions_exp[[common_column] + additional_columns], on=common_column, how="left")
    # print(gfc_df.dtypes)

    # Convert JSON geometry to Shapely objects
    gfc_df['geometry'] = gfc_df['geometry'].apply(lambda x: shape(json.loads(x)))
    # gfc_df.dtypes

    # Create GeoDataFrame from dataframe
    gdf_ecoregions = gpd.GeoDataFrame(gfc_df, geometry='geometry')
    print(gdf_ecoregions.dtypes)
    
    # # Plot
    # gdf_ecoregions.plot()
    # plt.show()

    # Visualize the deforestation rate for each ecoregion

    # plot using a grayscale colormap
    plot_driver(gdf_ecoregions, 'PrimaryLoss_Rate50')

    # Save the updated GeoDataFrame to a new geopackage
    gdf_ecoregions = gdf_ecoregions.set_crs(epsg=4326, inplace=True)
    path_gpkg = 'outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg'
    gdf_ecoregions.to_file(path_gpkg, layer="ZonalStat_Ecoregions", driver="GPKG")


if __name__ == "__main__":  
    run()
