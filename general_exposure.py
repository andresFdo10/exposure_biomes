import pandas as pd
import geopandas as gpd
import fiona
import os
import matplotlib.pyplot as plt

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

def plot_deforestation_trends(ecoregions_gdf, target_ecoregions):
    years = list(range(2001, 2024))

    plt.figure(figsize=(12, 6))

    for ecoregion in target_ecoregions:
        data = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        losses = [data[f"treecover_loss_{year}_50"].values[0] for year in years]

        plt.plot(years, losses, marker='o', label=ecoregion)

    plt.xlabel("Year")
    plt.ylabel("Tree Cover Loss (ha)")
    plt.title("Deforestation Trends (2001-2023)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_deforestation_rate(ecoregions_gdf, target_ecoregions):
    years = list(range(2002, 2024))

    plt.figure(figsize=(12, 6))

    for ecoregion in target_ecoregions:
        data = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        rates = [data[f"deforestation_rate_{year}_50"].values[0] for year in years]

        plt.bar(years, rates, alpha=0.6, label=ecoregion)

    plt.xlabel("Year")
    plt.ylabel("Deforestation Rate (%)")
    plt.title("Annual Change in Deforestation Rate (2002-2023)")
    plt.axhline(y=0, color='black', linestyle='--')  # Reference line at 0%
    plt.legend()
    plt.grid()
    plt.show()



def run():
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"

    if not os.path.exists(path_gpkg):
        raise FileNotFoundError(f"File not found: {path_gpkg}")

    # List all layers in the geopackage
    layers = fiona.listlayers(path_gpkg)
    print('Available layers: ', layers)

    # Read the geopackage using geopandas
    exposure_nino = gpd.read_file(path_gpkg, layer="ZonalStat_Ecoregions_EWM_nino_nina")
    print('Exposure El Nino')
    print(exposure_nino.head())
    print('\n')

    # Generate column names for tree cover loss from 2001 to 2023
    years = list(range(2001, 2024))
    new_columns = [f"treecover_loss_{year}_50" for year in years]

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
                column_name = f"treecover_loss_{year}_50"
                if column_name in exposure_nino.columns:
                    exposure_nino.loc[exposure_nino['ECO_NAME'] == ecoregion_name, column_name] = loss_value

            # Compute deforestation rates from 2002 onward
            for year in range(2002, 2024):
                prev_year = year - 1
                loss_current = loss_by_year.get(year, 0)
                loss_previous = loss_by_year.get(prev_year, 0)

                if loss_previous > 0:
                    rate = ((loss_current - loss_previous) / loss_previous) * 100 # Percentage change

                else:
                    rate = 0 # Avoid division by zero

                rate_column_name = f"deforestation_rate_{year}_50"
                exposure_nino.loc[exposure_nino['ECO_NAME'] == ecoregion_name, rate_column_name] = rate


    print('\nUpdated GeoDataFrame for Bahia coastal forests:')
    print(exposure_nino[exposure_nino['ECO_NAME'] == "Bahia coastal forests"])
    print(exposure_nino[exposure_nino['ECO_NAME'] == "Cauca Valley montane forests"])

    # Save the updated GeoDataFrame to a new geopackage
    output_layer_name = 'Ecoregions_EWM_nino_nina_GFW'
    exposure_nino.to_file(path_gpkg, layer=output_layer_name, driver="GPKG")
    print(f"GeoDataFrame saved to {path_gpkg} as layer {output_layer_name}")

    # Call the function after processing the data
    plot_deforestation_trends(exposure_nino, target_ecoregions)
    # Call the function
    plot_deforestation_rate(exposure_nino, target_ecoregions)

if __name__ == "__main__":
    run()
