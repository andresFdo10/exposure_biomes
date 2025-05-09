import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from scipy.stats import wilcoxon
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.crs import CRS
import os
def run():
    # Load the ecoregions GeoPackage
    gpkg_path = "inputs/01_geopackages/Neotropics.gpkg"
    layer_name = "ecoregions2017"
    ecoregions_gdf = gpd.read_file(gpkg_path, layer=layer_name)
    ecoregions_gdf.plot()
    plt.show()

    folder_path = "outputs/raster/"
    long_nino_raster = os.path.join(folder_path, "exposure_EWM_nino_60yrs.tif")
    after_nino_raster = os.path.join(folder_path, "exposure_EWM_nino_after.tif")
    long_nina_raster = os.path.join(folder_path, "exposure_EWM_nina_60yrs.tif")
    after_nina_raster = os.path.join(folder_path, "exposure_EWM_nina_after.tif")
    
    # Check and assgin CRS if missing
    target_crs = CRS.from_epsg(4326)
    for raster_file in [long_nino_raster, after_nino_raster, long_nina_raster, after_nino_raster]:
        with rasterio.open(raster_file, 'r+') as src:
            if src.crs is None:
                src.crs = target_crs
                print(f"{raster_file}: CRS assigned (EPSG:4326).")
            else:
                print(f"{raster_file}: CRS already assigned ({src.crs.to_string()}). Skipped.")

        
    # Zonal statistics with raster_out=True
    long_nino_stats = zonal_stats(ecoregions_gdf, long_nino_raster, stats=None, raster_out=True)
    after_nino_stats = zonal_stats(ecoregions_gdf, after_nino_raster, stats=None, raster_out=True)
    long_nina_stats = zonal_stats(ecoregions_gdf, long_nina_raster, stats=None, raster_out=True)
    after_nina_stats = zonal_stats(ecoregions_gdf, after_nina_raster, stats=None, raster_out=True)
    
    # Example statistical test loop (Wilcoxon per ecoregion)
    results = []
    for i, region in enumerate(ecoregions_gdf.itertuples()):
        long_nino = long_nino_stats[i]['mini_raster_array'].compressed()
        after_nino = after_nino_stats[i]['mini_raster_array'].compressed()

        valid = (~pd.isnull(long_nino)) & (~pd.isnull(after_nino))
        if valid.sum() >= 10:
            try:
                stat, p = wilcoxon(long_nino[valid], after_nino[valid])
                median_diff = pd.Series(after_nino[valid]).median() - pd.Series(long_nino[valid]).median()
            except ValueError:
                p, median_diff = None, None
        else:
            p, median_diff = None, None

        results.append({
            'ECO_ID': getattr(region, 'ECO_ID', None),
            'Ecoregion': getattr(region, 'NAME', f'Region_{i}'),
            'Wilcoxon_p': p,
            'Median_Diff': median_diff
        })

    # Save or review results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(folder_path, "ewi_el_nino_wilcoxon_results_long_ts.csv"), index=False)
    

    # Example statistical test loop (Wilcoxon per ecoregion)
    results = []
    for i, region in enumerate(ecoregions_gdf.itertuples()):
        long_nina = long_nina_stats[i]['mini_raster_array'].compressed()
        after_nina = after_nina_stats[i]['mini_raster_array'].compressed()

        valid = (~pd.isnull(long_nina)) & (~pd.isnull(after_nina))
        if valid.sum() >= 10:
            try:
                stat, p = wilcoxon(long_nina[valid], after_nina[valid])
                median_diff = pd.Series(after_nina[valid]).median() - pd.Series(long_nina[valid]).median()
            except ValueError:
                p, median_diff = None, None
        else:
            p, median_diff = None, None

        results.append({
            'ECO_ID': getattr(region, 'ECO_ID', None),
            'Ecoregion': getattr(region, 'NAME', f'Region_{i}'),
            'Wilcoxon_p': p,
            'Median_Diff': median_diff
        })

    # Save or review results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(folder_path, "ewi_el_nina_wilcoxon_results_long_ts.csv"), index=False)

if __name__ == "__main__":
    run()