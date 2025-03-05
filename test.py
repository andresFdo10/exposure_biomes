import pandas as pd
import geopandas as gpd
import fiona
import os
import matplotlib.pyplot as plt

def load_gfw_data(ecoregion, threshold, data):
    data_name = data  # Use = for assignment, not == for comparison
    if data_name == "treecover_extent":
        path_name = f"./inputs/04_csv/{ecoregion}/treecover_{threshold}/Primary forest in {ecoregion}/"
        data_name = "treecover_extent_2000__ha.csv"
    elif data_name == "treecover_loss":
        path_name = f"./inputs/04_csv/{ecoregion}/treecover_{threshold}/Primary forest loss in {ecoregion}/"
        data_name = "treecover_loss__ha.csv"
    elif data_name == "primary_forest":
        path_name = f"./inputs/04_csv/{ecoregion}/treecover_{threshold}/Primary forest in {ecoregion}/"
        data_name = "treecover_extent_in_primary_forests_2001_tropics_only__ha.csv"
    elif data_name == "primary_forest_loss":
        path_name = f"./inputs/04_csv/{ecoregion}/treecover_{threshold}/Primary forest loss in {ecoregion}/"
        data_name = "treecover_loss_in_primary_forests_2001_tropics_only__ha.csv"

    gfw = pd.read_csv(f"{path_name}{data_name}")
    gfw['Ecoregion'] = ecoregion
def compute_absolute_deforestation_rate(ecoregions_gdf, target_ecoregions):
    """ Compute absolute deforestation rate for each ecoregion based on total tree cover loss. """
    for ecoregion in target_ecoregions:
        data = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        if not data.empty:
            # Load initial tree cover dynamically
            ecoregion_extent = load_gfw_data(ecoregion, 50, "treecover_extent")
            if "umd_tree_cover_extent_2000__ha" in ecoregion_extent.columns:
                initial_tree_cover = ecoregion_extent["umd_tree_cover_extent_2000__ha"].iloc[0]
            else:
                initial_tree_cover = None

            # Compute total loss from 2001 to 2023
            total_loss = sum(data[f"treecover_loss_{year}_50"].values[0] for year in range(2001, 2024))
            
            # Compute absolute rate
            absolute_rate = (total_loss / initial_tree_cover) * 100 if initial_tree_cover and initial_tree_cover > 0 else None
            
            # Store the rate in the GeoDataFrame
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "absolute_deforestation_rate"] = absolute_rate
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "total_treecover_loss"] = total_loss

def compute_normalized_deforestation_velocity(ecoregions_gdf, target_ecoregions, years):
    """Computes the annual deforestation velocity (ha/year) and normalizes it by initial tree cover."""
    velocities = {}
    normalized_rates = {}

    for ecoregion in target_ecoregions:
        data = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        if not data.empty:
            total_loss = sum(data[f"treecover_loss_{year}_50"].values[0] for year in years)
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
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "deforestation_velocity_ha_per_year"] = velocity
            ecoregions_gdf.loc[ecoregions_gdf['ECO_NAME'] == ecoregion, "normalized_deforestation_rate_percent_per_year"] = normalized_rate

    return velocities, normalized_rates

def plot_deforestation_rate(ecoregions_gdf, target_ecoregions):
    years = list(range(2002, 2024))

    plt.figure(figsize=(12, 6))

    for ecoregion in target_ecoregions:
        data = ecoregions_gdf[ecoregions_gdf['ECO_NAME'] == ecoregion]
        if data.empty:
            print(f"Warning: No data found for {ecoregion}. Skipping...")
            continue
        
        rates = [data[f"deforestation_rate_{year}_50"].values[0] if f"deforestation_rate_{year}_50" in data.columns else 0 for year in years]

        plt.bar(years, rates, alpha=0.6, label=ecoregion)

    plt.xlabel("Year")
    plt.ylabel("Deforestation Rate (%)")
    plt.title("Annual Change in Deforestation Rate (2002-2023)")
    plt.axhline(y=0, color='black', linestyle='--')  # Reference line at 0%
    plt.legend()
    plt.grid()
    plt.show()

def plot_normalized_deforestation_velocity(normalized_rates):
    """Plots a bar chart comparing normalized deforestation 
    velocity (percentage of initial tree cover lost per year)."""
    ecoregions = list(normalized_rates.keys())
    values = list(normalized_rates.values())

    plt.figure(figsize=(12, 6))
    plt.bar(ecoregions, values, color='orange')
    plt.xlabel("Ecoregions")
    plt.ylabel("Deforestation Velocity (% of Initial Tree Cover / year)")
    plt.title("Comparison of Normalized Annual Deforestation Rate by Ecoregion (2001-2023)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.show()

def run():
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"
    
    if not os.path.exists(path_gpkg):
        raise FileNotFoundError(f"File not found: {path_gpkg}")

    layers = fiona.listlayers(path_gpkg)
    print('Available layers: ', layers)

    exposure_nino = gpd.read_file(path_gpkg, layer="ZonalStat_Ecoregions_EWM_nino_nina")
    
    years = list(range(2001, 2024))
    new_columns = [f"treecover_loss_{year}_50" for year in years]
    rate_columns = [f"deforestation_rate_{year}_50" for year in range(2002, 2024)]

    for column in new_columns + rate_columns:
        exposure_nino[column] = float('nan')
    
    target_ecoregions = [
        'Bahia coastal forests', 
        'Cauca Valley montane forests', 
        'Cordillera de Merida páramo',
        'Cordillera Oriental montane forests',
        'Magdalena Valley montane forests',
        'Venezuelan Andes montane forests'
    ]

    velocities, normalized_rates = compute_normalized_deforestation_velocity(exposure_nino, target_ecoregions, years)
    plot_normalized_deforestation_velocity(normalized_rates)
    plot_deforestation_rate(exposure_nino, target_ecoregions)
    plot_normalized_deforestation_velocity

if __name__ == "__main__":
    run()
