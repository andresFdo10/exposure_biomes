import pandas as pd
import geopandas as gpd
import fiona
import os

def gfw_data(ecoregion, threshold, data):
    data_name = data  # Use = for assignment, not == for comparison
    if data_name == "treecover_extent":
        data_name = "treecover_extent_2000_in_primary_forests_2001_tropics_only__ha.csv"
    elif data_name == "treecover_loss":
        data_name = "treecover_loss__ha.csv"
    elif data_name == "loss_primary":
        data_name = "treecover_loss_in_primary_forests_2001_tropics_only__ha.csv"

    gfw = pd.read_csv(f"inputs/04_csv/{ecoregion}/tree_cover_{threshold}/Primary Forest loss in {ecoregion} {threshold}/{data_name}")

    return gfw

def run():
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"

    if not os.path.exists(path_gpkg):
        raise FileNotFoundError(f"File not found: {path_gpkg}")

    # List all layers in the geopackage
    layers = fiona.listlayers(path_gpkg)

    exposure_nino = gpd.read_file(path_gpkg, layer="ZonalStat_Ecoregions_EWM_nino_nina")

    bahia = gfw_data("Bahia coastal forests", 50, "treecover_extent")
    bahia_loss = gfw_data("Bahia coastal forests", 30, "treecover_loss")
    bahia_loss_primary = gfw_data("Bahia coastal forests", 50, "loss_primary")
    print(bahia_loss_primary)

if __name__ == "__main__":
    run()
