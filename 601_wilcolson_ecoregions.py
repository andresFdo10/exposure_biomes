import geopandas as gpd
import pandas as pd
def run():
    # Load the GeoPackage
    gdf = gpd.read_file("outputs/geopackages/EWI_before_after.gpkg", layer="ecoregions2017_wicolxon")
    # print(gdf.columns)

    # rename columns
    gdf = gdf.rename(columns={
        'ewi_el_nino_wilcoxon_results_Wilcoxon_p': 'wicolxon_p_nino',
        'ewi_el_nina_wilcoxon_results_Wilcoxon_p': 'wicolxon_p_nina',
        'ewi_el_nino_wilcoxon_results_Median_Diff': 'Delta_EWI_nino',
        'ewi_el_nina_wilcoxon_results_Median_Diff': 'Delta_EWI_nina'
    })
    print(gdf.dtypes)

    # Check column type (optional)
    print(gdf['Delta_EWI_nino'].dtype)

    # Convert from string to float
    gdf['Delta_EWI_nino'] = pd.to_numeric(gdf['Delta_EWI_nino'], errors='coerce')
    gdf['Delta_EWI_nina'] = pd.to_numeric(gdf['Delta_EWI_nina'], errors='coerce')
    gdf['wicolxon_p_nino'] = pd.to_numeric(gdf['wicolxon_p_nino'], errors='coerce')
    gdf['wicolxon_p_nina'] = pd.to_numeric(gdf['wicolxon_p_nina'], errors='coerce')
    print(gdf.describe())

    # # Optional: check conversion
    # print(gdf['Delta_EWI'].describe())

    # Save the updated GeoPackage (optional)
    gdf.to_file("outputs/geopackages/EWI_before_after.gpkg", layer="EWI_before_after_wicolxon_test", driver="GPKG")

if __name__ == "__main__":
    run()