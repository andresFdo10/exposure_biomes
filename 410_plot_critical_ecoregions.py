import pandas as pd
import geopandas as gpd



def run():
    
    # Step 1: Load the original Geopackage with ecoregions
    path_gpkg = 'outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg'
    layer_name = 'ZonalStat_Ecoregions'
    ecoregions_gdf = gpd.read_file(path_gpkg, layer=layer_name)
    print(f"Original dataset: \n{ecoregions_gdf.dtypes}")

    # Step 2: Load the PCA CSV with the filtered ecoregions 
    critical_ecoregions = pd.read_csv('outputs/csv/ecoregions_combined_effects.csv')
    print(f"\nCritical ecoregions: \n{critical_ecoregions.dtypes}")

    # Step 3: Merge using "ECO_ID" (keeping all ecoregions)
    merged_gdf = ecoregions_gdf.merge(critical_ecoregions, on="ECO_ID", how="left")
    print(f"\nMerged GDF:\n {merged_gdf.dtypes}")

    # Step 4: Save as a new Geopackage file
    output_gpkg = "outputs/geopackages/critical_ecoregions.gpkg"
    merged_gdf.to_file(output_gpkg, driver="GPKG", layer="critical_ecoregions")

if __name__ == "__main__":
    run()