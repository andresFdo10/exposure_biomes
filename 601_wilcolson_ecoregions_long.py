import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
def run():
    # Load the GeoPackage
    gdf = gpd.read_file("inputs/01_geopackages/Neotropics.gpkg", layer="ecoregions2017")
    # print(gdf.columns)

    # Load csv with wicolxon nino
    df_wicolxon_nino = pd.read_csv("outputs/raster/ewi_el_nino_wilcoxon_results.csv")
    print(df_wicolxon_nino.dtypes)
    df_wicolxon_nina = pd.read_csv("outputs/raster/ewi_el_nina_wilcoxon_results.csv")
    print(df_wicolxon_nina.dtypes)

    # Merge the dataframes
    gdf = gdf.merge(df_wicolxon_nino, on='ECO_ID', how='left')
    gdf = gdf.rename(columns={'Wilcoxon_p': 'Wicolxon_p_nino', 'Median_Diff': 'Delta_EWI_nino'})
    gdf = gdf.merge(df_wicolxon_nina, on='ECO_ID', how='left')
    gdf = gdf.rename(columns={'Wilcoxon_p': 'Wicolxon_p_nina', 'Median_Diff': 'Delta_EWI_nina'})
    print(gdf.dtypes)

    # Filtering by p-value < 0.01
    gdf_nino = gdf[gdf['Wicolxon_p_nino'] < 0.01]
    gdf_nina = gdf[gdf['Wicolxon_p_nina'] < 0.01]

    min_val = min(gdf_nino['Delta_EWI_nino'].min(), gdf_nina['Delta_EWI_nina'].min())
    max_val = max(gdf_nino['Delta_EWI_nino'].max(), gdf_nina['Delta_EWI_nina'].max())
    


        # Set up a 1x2 plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot ΔEWI El Niño
    gdf_nino.plot(column='Delta_EWI_nino', cmap='BrBG', legend=True,
            legend_kwds={'label': "ΔEWI El Niño"}, vmin=min_val, vmax=max_val, ax=axes[0])
    axes[0].set_title("ΔEWI El Niño (1991–2021 vs. 1961–2021)")
    axes[0].axis("off")

    # Plot ΔEWI La Niña
    gdf_nina.plot(column='Delta_EWI_nina', cmap='BrBG', legend=True,
            legend_kwds={'label': "ΔEWI La Niña"}, vmin=min_val, vmax=max_val, ax=axes[1])
    axes[1].set_title("ΔEWI La Niña (1991–2021 vs. 1961–2021)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/figures/fig_wicolxon_test.png")
    plt.show()

    # gdf.to_file("outputs/geopackages/EWI_before_after.gpkg", layer="EWI_before_after_wicolxon_test", driver="GPKG")

if __name__ == "__main__":
    run()