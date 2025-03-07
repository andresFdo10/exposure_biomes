import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt

def run():
    # Read the geopackage file
    path_gpkg = 'outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg'
    if not os.path.exists(path_gpkg):
        raise FileNotFoundError(f"File not found: {path_gpkg}")

    # Read the geopackage using geopandas
    ecoregions_exp = gpd.read_file(path_gpkg, layer="zonal_statistics")
    ecoregions_exp.plot()
    plt.show()
    print("ecoregions_exp/n",ecoregions_exp.columns)

    forest_cover = pd.read_csv('inputs/04_csv/Forest_Loss_Per_Year_Stats_50.csv')
    forest_cover = forest_cover.drop(['.geo', 'system:index'], axis=1)

    # forest_cover['Loss_rate'] = ((forest_cover['LossArea_50'] / forest_cover['ForestArea_50']) * 23 ) *  100
    # forest_cover.to_csv('outputs/csv/Forest_Zonal_Stats_50.csv', index=False)
    print(forest_cover.columns)
    print(forest_cover['Loss_Per_Year_%'])

    # Identify the common column
    common_column = 'ECO_NAME'

    # Merge the DataFrames based on the common column
    # Select only the two additional columns from CSV
    additional_columns = [col for col in forest_cover.columns if col not in ecoregions_exp.columns and col != common_column]
    # Merge the DataFrame with the GeoDataFrame
    ecoregions_exp = ecoregions_exp.merge(forest_cover[[common_column] + additional_columns], on=common_column, how="left")
    print(ecoregions_exp.columns)

    # Save the updated GeoDataFrame to a new geopackage
    ecoregions_exp.to_file('outputs/geopackages/ZonalStat_Ecoregions_EWM_v2.gpkg', driver='GPKG', layer='zonal_statistics_v2')




if __name__ == "__main__":
    run()