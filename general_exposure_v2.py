import pandas as pd
import geopandas as gpd
import fiona
import os
import matplotlib.pyplot as plt

def load_gfw_data(ecoregion, threshold, data):
    data_name = data  # Use = for assignment, not == for comparison
    path_name = f"./inputs/04_csv/{ecoregion}/treecover_{threshold}/"  # Default path

    if data_name == "treecover_extent":
        data_name = "treecover_extent_2000__ha.csv"
    elif data_name == "treecover_loss":
        data_name = "treecover_loss__ha.csv"
    elif data_name == "primary_forest":
        data_name = "treecover_extent_2000_in_primary_forests_2001_tropics_only__ha.csv"
    elif data_name == "primary_forest_loss":
        data_name = "treecover_loss_in_primary_forests_2001_tropics_only__ha.csv"
    elif data_name == "treecover_loss_by_fires":  # Correct the condition
        data_name = "treecover_loss_from_fires_by_region__ha.csv"
    else:
        raise ValueError(f"Unsupported data type '{data}' passed to load_gfw_data()")

    file_path = f"{path_name}{data_name}"

    # Ensure file exists before reading
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    gfw = pd.read_csv(file_path)
    gfw['Ecoregion'] = ecoregion

    return gfw

def compute_relative_deforestation_rate(ecoregions_gdf, target_ecoregions):
    """ Compute relative deforestation rate for each ecoregion based on total tree cover loss. """
    for ecoregion in target_ecoregions:
        data = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        if not data.empty:
            # Load tree cover loss data
            ecoregion_loss = load_gfw_data(ecoregion, 50, "treecover_loss")
            if 'umd_tree_cover_loss__year' in ecoregion_loss.columns and 'umd_tree_cover_loss__ha' in ecoregion_loss.columns:
                loss_by_year = dict(zip(ecoregion_loss['umd_tree_cover_loss__year'], ecoregion_loss['umd_tree_cover_loss__ha']))
                for year, loss_value in loss_by_year.items():
                    column_name = f"tc_loss_{year}_50"
                    if column_name in ecoregions_gdf.columns:
                        ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, column_name] = loss_value

            # Compute total loss from 2001 to 2023
            total_loss = sum(loss_by_year.get(year, 0) for year in range(2001, 2024))
            
            # Load initial tree cover dynamically
            ecoregion_extent = load_gfw_data(ecoregion, 50, "treecover_extent")
            if "umd_tree_cover_extent_2000__ha" in ecoregion_extent.columns:
                initial_tree_cover = ecoregion_extent["umd_tree_cover_extent_2000__ha"].iloc[0]
            else:
                initial_tree_cover = None
            
            # Compute relative rate
            relative_rate = (total_loss / initial_tree_cover) * 100 if initial_tree_cover and initial_tree_cover > 0 else None
            
            # Store the rate in the GeoDataFrame
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "relative_loss_rate"] = relative_rate
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "total_loss"] = total_loss
            
        # Primary forest loss
        primary_forest_loss = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        if not primary_forest_loss.empty:
            # Load tree cover loss data
            primary_loss = load_gfw_data(ecoregion, 50, "primary_forest_loss")
            if 'umd_tree_cover_loss__year' in primary_loss.columns and 'umd_tree_cover_loss__ha' in primary_loss.columns:
                loss_by_year = dict(zip(primary_loss['umd_tree_cover_loss__year'], primary_loss['umd_tree_cover_loss__ha']))
                for year, loss_value in loss_by_year.items():
                    column_name = f"primary_loss_{year}_50"
                    if column_name in ecoregions_gdf.columns:
                        ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, column_name] = loss_value

            # Compute total loss from 2001 to 2023
            total_loss = sum(loss_by_year.get(year, 0) for year in range(2001, 2024))

            # Load initial tree cover dynamically
            primary_extent = load_gfw_data(ecoregion, 50, "primary_forest")
            if "umd_tree_cover_extent_2000__ha" in primary_extent.columns:
                initial_primary = primary_extent["umd_tree_cover_extent_2000__ha"].iloc[0]
            else:
                initial_primary = None
            
            # Compute relative rate
            relative_rate = (total_loss / initial_primary) * 100 if initial_primary and initial_primary > 0 else None
            
            # Store the rate in the GeoDataFrame
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "relative_primaryLoss_rate"] = relative_rate
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "total_primary_loss"] = total_loss

        # Treecover loss by fires
        loss_by_fires = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        if not loss_by_fires.empty:
            # Load tree cover loss data
            tc_loss_by_fires = load_gfw_data(ecoregion, 50, "treecover_loss_by_fires")

            if 'umd_tree_cover_loss__year' in tc_loss_by_fires.columns and 'umd_tree_cover_loss_from_fires__ha' in tc_loss_by_fires.columns:
                # FIX: Use correct mapping between year and loss
                loss_by_year = dict(zip(tc_loss_by_fires['umd_tree_cover_loss__year'], tc_loss_by_fires['umd_tree_cover_loss_from_fires__ha']))

                for year, loss_value in loss_by_year.items():
                    column_name = f"tc_loss_by_fires_{year}_50"
                    if column_name in ecoregions_gdf.columns:
                        ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, column_name] = loss_value

                # Compute total loss from 2001 to 2023
                total_loss = sum(loss_by_year.get(year, 0) for year in range(2001, 2024))

                ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "total_tc_loss_by_fires"] = total_loss




    return ecoregions_gdf

def compute_normalized_deforestation_velocity(ecoregions_gdf, target_ecoregions, years):
    """Computes the annual deforestation velocity (ha/year) and normalizes it by initial tree cover."""
    velocities = {}
    normalized_rates = {}

    for ecoregion in target_ecoregions:
        data = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        if not data.empty:
            total_loss = sum(data[f"tc_loss_{year}_50"].values[0] for year in years if f"tc_loss_{year}_50" in data.columns)
            num_years = len(years)
            velocity = total_loss / num_years  # ha/year

            # Load initial tree cover dynamically
            ecoregion_extent = load_gfw_data(ecoregion, 50, "treecover_extent")
            if "umd_tree_cover_extent_2000__ha" in ecoregion_extent.columns:
                initial_tree_cover = ecoregion_extent["umd_tree_cover_extent_2000__ha"].iloc[0]
            else:
                initial_tree_cover = None

            # Compute normalized rate (percentage of initial tree cover per year)
            normalized_rate = (velocity / initial_tree_cover) * 100 if initial_tree_cover and initial_tree_cover > 0 else None

            velocities[ecoregion] = velocity
            normalized_rates[ecoregion] = normalized_rate

            # Store values in GeoDataFrame
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "tc_loss_velocity"] = velocity
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "norm_tc_loss_rate"] = normalized_rate

    return ecoregions_gdf

def compute_normalized_primaryLoss_velocity(ecoregions_gdf, target_ecoregions, years):
    """Computes the annual deforestation velocity (ha/year) and normalizes it by initial tree cover."""
    velocities = {}
    normalized_rates = {}

    for ecoregion in target_ecoregions:
        data = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        if not data.empty:
            total_loss = sum(data[f"primary_loss_{year}_50"].values[0] for year in years if f"primary_loss_{year}_50" in data.columns)
            num_years = len(years)
            velocity = total_loss / num_years  # ha/year

            # Load initial tree cover dynamically
            ecoregion_extent = load_gfw_data(ecoregion, 50, "primary_forest")
            if "umd_tree_cover_extent_2000__ha" in ecoregion_extent.columns:
                initial_tree_cover = ecoregion_extent["umd_tree_cover_extent_2000__ha"].iloc[0]
            else:
                initial_tree_cover = None

            # Compute normalized rate (percentage of initial tree cover per year)
            normalized_rate = (velocity / initial_tree_cover) * 100 if initial_tree_cover and initial_tree_cover > 0 else None

            velocities[ecoregion] = velocity
            normalized_rates[ecoregion] = normalized_rate

            # Store values in GeoDataFrame
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "primary_loss_velocity"] = velocity
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "norm_primary_loss_rate"] = normalized_rate

    return ecoregions_gdf


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
    new_columns = [f"tc_loss_{year}_50" for year in years]

    # Add new columns to the GeoDataFrame with default values 0
    for column in new_columns:
        exposure_nino[column] = float('nan')


    new_columns2 = [f"primary_loss_{year}_50" for year in years]
    # Add new columns to the GeoDataFrame with default values 0
    for column in new_columns2:
        exposure_nino[column] = float('nan')

    new_columns3 = [f"tc_loss_by_fires_{year}_50" for year in years]
    # Add new columns to the GeoDataFrame with default values 0
    for column in new_columns3:
        exposure_nino[column] = float('nan')

    # Define targe ecoregions
    target_ecoregions = [
        'Bahia coastal forests', 
        'Cauca valley montane forests', 
        'Cordillera de Merida páramo',
        'Cordillera Oriental montane forests',
        'Magdalena Valley montane forests',
        'Guianan lowland moist forests',
        # 'Uatumã-Trombetas moist forests',
        'Tapajós-Xingu moist forests',
        'Xingu-Tocantins-Araguaia moist forests',
        'Guianan Highlands moist forests',
        'Ucayali moist forests',
        'Iquitos várzea',
        'Guianan savanna',
        'Magdalena-Urabá moist forests',
        'Chocó-Darién moist forests',
        'Apure-Villavicencio dry forests',
        # 'Monte Alegre várzea',
        # 'Guajira-Barranquilla xeric scrub',
        'Maracaibo dry forests',
        'Catatumbo moist forests',
        'Magdalena Valley dry forests',
        'Lara-Falcón dry forests',
        'Paraguaná xeric scrub',
        # 'Cordillera La Costa montane forests',
        'Cauca Valley dry forests',
        # 'Eastern Panamanian montane forests',
        'Patía valley dry forests',
        'Venezuelan Andes montane forests'
        ]
    # Define the years range
    years = list(range(2001, 2024))

    print('\nUpdated GeoDataFrame for Bahia coastal forests:')

        
    gdf_ecoregions = compute_relative_deforestation_rate(exposure_nino, target_ecoregions)
    print(gdf_ecoregions[gdf_ecoregions['ECO_NAME'] == "Bahia coastal forests"].columns)

    # Compute velocities and normalized rates for each ecoregion
    gdf_ecoregions = compute_normalized_deforestation_velocity(gdf_ecoregions, target_ecoregions, years)
    gdf_ecoregions = compute_normalized_primaryLoss_velocity(gdf_ecoregions, target_ecoregions, years)
    gdf_ecoregions['perc_loss_fires'] = (gdf_ecoregions['total_tc_loss_by_fires'] / gdf_ecoregions['total_loss']) * 100
    print(gdf_ecoregions.columns)

    # Save results into the GeoPackage
    output_layer_name = "Ecoregions_Loss"
    gdf_ecoregions.to_file(path_gpkg, layer=output_layer_name, driver="GPKG")

    

if __name__ == "__main__":
    run()
