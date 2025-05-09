import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

def run():

    # Load your GeoPackage (replace with actual path if needed)
    gdf = gpd.read_file("outputs/geopackages/EWI_before_after.gpkg", layer="EWI_before_after_wicolxon_test")

    # Ensure the delta columns are numeric
    gdf['Delta_EWI_nino'] = pd.to_numeric(gdf['Delta_EWI_nino'], errors='coerce')
    gdf['Delta_EWI_nina'] = pd.to_numeric(gdf['Delta_EWI_nina'], errors='coerce')

    # Set up a 1x2 plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot ΔEWI El Niño
    gdf.plot(column='Delta_EWI_nino', cmap='RdBu_r', legend=True,
            legend_kwds={'label': "ΔEWI El Niño"}, ax=axes[0])
    axes[0].set_title("ΔEWI El Niño (1991–2021 vs. 1961–1990)")
    axes[0].axis("off")

    # Plot ΔEWI La Niña
    gdf.plot(column='Delta_EWI_nina', cmap='RdBu_r', legend=True,
            legend_kwds={'label': "ΔEWI La Niña"}, ax=axes[1])
    axes[1].set_title("ΔEWI La Niña (1991–2021 vs. 1961–1990)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()