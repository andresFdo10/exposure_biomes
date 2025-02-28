import pandas as pd
import geopandas as gpd
import fiona
import os

def load_gfw_data(ecoregion, threshold, data):
    data_name = data  # Use = for assignment, not == for comparison
    if data_name == "treecover_extent":
        data_name = "treecover_extent_2000_in_primary_forests_2001_tropics_only__ha.csv"
    elif data_name == "treecover_loss":
        data_name = "treecover_loss__ha.csv"
    elif data_name == "loss_primary":
        data_name = "treecover_loss_in_primary_forests_2001_tropics_only__ha.csv"

    gfw = pd.read_csv(f"inputs/04_csv/{ecoregion}/treecover_{threshold}/Primary Forest loss in {ecoregion}/{data_name}")
    gfw['Ecoregion'] = ecoregion

    return gfw

def run():
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"

    if not os.path.exists(path_gpkg):
        raise FileNotFoundError(f"File not found: {path_gpkg}")

    # List all layers in the geopackage
    layers = fiona.listlayers(path_gpkg)
    print('Available layers: ', layers)

    exposure_nino = gpd.read_file(path_gpkg, layer="ZonalStat_Ecoregions_EWM_nino_nina")
    print('Exposure El Nino')
    print(exposure_nino.head())
    print('\n')

    # Generate column names for tree cover loss from 2001 to 2023
    years = list(range(2001, 2024))
    new_columns = [f"treecover_loss_{year}" for year in years]

    # Add new columns to the GeoDataFrame with default values 0
    for column in new_columns:
        exposure_nino[column] = float('nan')

    print('\nGeoDataFrame with new columns:')
    print(exposure_nino.head())
    print('\n')
    # Define targe ecoregions
    target_ecoregions = [
        'Bahia coastal forests', 
        'Cauca Valley montane forests', 
        'Cordillera de Merida páramo',
        'Cordillera Oriental montane forests',
        'Magdalena Valley montane forests',
        'Venezuelan Andes montane forests'
        ]

    # bahia_loss = load_gfw_data("Bahia coastal forests", 50, "treecover_loss")

    # Iterate over each target ecoregion and populate tree cover loss columns
    for ecoregion_name in target_ecoregions:
        ecoregion_loss = load_gfw_data(ecoregion_name, 50, "treecover_loss")
        # Extract tree cover loss values from the CSV
        # Assuming the CSV has columns 'year' and 'umd_tree_cover_loss__ha'
        if 'umd_tree_cover_loss__year' in ecoregion_loss.columns and 'umd_tree_cover_loss__ha' in ecoregion_loss.columns:
            loss_by_year = dict(zip(ecoregion_loss['umd_tree_cover_loss__year'], ecoregion_loss['umd_tree_cover_loss__ha']))
            print(f'Loss by Year for {ecoregion_name}:')
            print(loss_by_year)
            # Update the GeoDataFrame for the "Bahia coastal forests" row
            # ecoregion_name = "Bahia coastal forests"
            
            for year, loss_value in loss_by_year.items():
                column_name = f"treecover_loss_{year}"
                if column_name in exposure_nino.columns:
                    exposure_nino.loc[exposure_nino['ECO_NAME'] == ecoregion_name, column_name] = loss_value

    print('\nUpdated GeoDataFrame for Bahia coastal forests:')
    print(exposure_nino[exposure_nino['ECO_NAME'] == "Bahia coastal forests"])
    print(exposure_nino[exposure_nino['ECO_NAME'] == "Cauca Valley montane forests"])

if __name__ == "__main__":
    run()
