import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

def run():
    # Load the ecoregions Geopackage
    gpkg_path = "inputs/01_geopackages/Neotropics.gpkg"
    layer_name = "ecoregions2017"
    ecoregions_gdf = gpd.read_file(gpkg_path, layer=layer_name)


    # Load the CSVs files with the ewi_el_nino and ewi_el_nina data
    ewi_nino_df = pd.read_csv("outputs/csv/ewi_nino_wilcoxon.csv")
    ewi_nino_df = ewi_nino_df[['ECO_ID', 'EWI_nino_after', 'EWI_nino_before']]
    ewi_nina_df = pd.read_csv("outputs/csv/ewi_nina_wilcoxon.csv")
    ewi_nina_df = ewi_nina_df[['ECO_ID', 'EWI_nina_after', 'EWI_nina_before']]
    ewi_nino_long_df = pd.read_csv("outputs/csv/ewi_nino_wilcoxon_long.csv")
    ewi_nino_long_df = ewi_nino_long_df[['ECO_ID', 'EWI_long_nino']]
    ewi_nina_long_df = pd.read_csv("outputs/csv/ewi_nina_wilcoxon_long.csv")
    ewi_nina_long_df = ewi_nina_long_df[['ECO_ID', 'EWI_long_nina']]

    # Merge the dataframes
    ecoregions_gdf = ecoregions_gdf.merge(ewi_nino_df, on='ECO_ID', how='left')
    ecoregions_gdf = ecoregions_gdf.merge(ewi_nina_df, on='ECO_ID', how='left')
    ecoregions_gdf = ecoregions_gdf.merge(ewi_nino_long_df, on='ECO_ID', how='left')
    ecoregions_gdf = ecoregions_gdf.merge(ewi_nina_long_df, on='ECO_ID', how='left')
    ecoregions_gdf["EWI_SD_nino"] = ecoregions_gdf[["EWI_nino_after", "EWI_nino_before", "EWI_long_nino"]].std(axis=1)
    ecoregions_gdf["EWI_SD_nina"] = ecoregions_gdf[["EWI_nina_after", "EWI_nina_before", "EWI_long_nina"]].std(axis=1)
    ecoregions_gdf.to_file("outputs/geopackages/EWI_before_after.gpkg", layer="EWI_standard_deviation", driver="GPKG")

    # plot the EWI_SD for el nino and la nina in a plot (1x2)using matplotlib
    vmin = min(ecoregions_gdf['EWI_SD_nino'].min(), ecoregions_gdf['EWI_SD_nina'].min())
    vmax = max(ecoregions_gdf['EWI_SD_nino'].max(), ecoregions_gdf['EWI_SD_nina'].max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ecoregions_gdf.plot(column='EWI_SD_nino', cmap='gray_r', vmin=vmin, vmax=vmax, ax=axes[0], legend=True)
    ecoregions_gdf.plot(column='EWI_SD_nina', cmap='gray_r', vmin=vmin, vmax=vmax, ax=axes[1], legend=True)
    axes[0].set_title('EWI SD nino (+/-)')
    axes[1].set_title('EWI SD nina (+/-)')
    plt.tight_layout()
    plt.show()
    


    # print dataframe
    print(ecoregions_gdf.head())



if __name__ == "__main__":
    run()