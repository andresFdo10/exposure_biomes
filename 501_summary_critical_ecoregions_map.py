
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import numpy as np

def add_features(ax, extent  = (-105, -33.9, -31.5, 23.8)):
    # Add coastlines
    ax.coastlines()

    # Add country borders with a dotted line style
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    # Set extent for dimension standardization
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add land features with an edge color
    land = cfeature.NaturalEarthFeature("physical", "land", "50m", facecolor="none")
    ax.add_feature(land, edgecolor="black", linewidth=0.2)
    # Add a scale bar
    # ax.add_scalebar(100) 
    # ax.add_feature(cfeature.NorthArrow(), loc='upper right')

    # Add gridlines with specified transparency and line width
    # ✅ Correct way to control gridline labels
    gl = ax.gridlines(alpha=0.6, linewidth=0.2, draw_labels=True)
    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = True
    # ✅ Set decimal degree format
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(-105, -30, 15))
    # gl.ylocator = mticker.FixedLocator(np.arange(-30, 30, 10))
    # gl.xformatter = mticker.FormatStrFormatter("%.1f")
    # gl.yformatter = mticker.FormatStrFormatter("%.1f")
    gl.xlabel_style = {'size': 9, 'color': 'black'}
    gl.ylabel_style = {'size': 9, 'color': 'black', 'rotation': 90}
    # Rotate right-side labels and set smaller font
    # ✅ Trigger rendering to populate label artists
    plt.draw()

def run():
    path_gpkg = "outputs/geopackages/ZonalStat_Ecoregions_EWM.gpkg"
    layer_name = "ZonalStat_Ecoregions"
    gdf = gpd.read_file(path_gpkg, layer=layer_name)

    # Extract just ECO_ID and ECO_NAME from the GeoDataFrame
    eco_names = gdf[["ECO_ID", "ECO_NAME"]]
    print(eco_names.head())

    # Load the csv file
    df = pd.read_csv('outputs/csv/ecoregions_combined_effects.csv') 
    print(df.head())

    # Assign a marker
    df['mark'] = 'X'

    # Pivot without aggregation
    pivot_df = df.pivot(index="ECO_ID", columns="Combined_Effect", values="mark").fillna('')

    # Reset index
    pivot_df.reset_index(inplace=True)

    # Ensure consistent column order
    pc_columns = ['PC1 & PC2', 'PC1 & PC3', 'PC2 & PC3']
    for col in pc_columns:
        if col not in pivot_df.columns:
            pivot_df[col] = ''

    # Reorder columns explicitly
    pivot_df = pivot_df[['ECO_ID'] + pc_columns]

    # Count combinations and define sorting key based on this preferred order
    pivot_df['combo_count'] = pivot_df[pc_columns].apply(lambda row: sum(val == 'X' for val in row), axis=1)

    def combination_code(row):
        return ''.join(['1' if row[col] == 'X' else '0' for col in pc_columns])

    pivot_df['sort_key'] = pivot_df.apply(combination_code, axis=1)

    # Sort by number of combinations (ascending), then by pattern
    pivot_df.sort_values(by=['combo_count', 'sort_key'], inplace=True)

    # Drop helper columns
    pivot_df.drop(columns=['combo_count', 'sort_key'], inplace=True)


    # Merge with summary table using ECO_ID
    merged_df = pd.merge(pivot_df, eco_names, on="ECO_ID", how="left")

    # Optional: reorder columns so ECO_NAME appears next to ECO_ID
    col_order = ['ECO_ID', 'ECO_NAME', 'PC1 & PC2', 'PC1 & PC3', 'PC2 & PC3']
    merged_df = merged_df[col_order]
    print(merged_df)


        # --- Filter only ecoregions with X in all 3 combinations ---
    target_ecos = merged_df[
        (merged_df['PC1 & PC2'] == 'X') &
        (merged_df['PC1 & PC3'] == 'X') &
        (merged_df['PC2 & PC3'] == 'X')
    ]['ECO_ID'].tolist()

    # Load full geodata again and filter
    critical_gdf = gdf[gdf["ECO_ID"].isin(target_ecos)]
    extent  = (-100, -33.9, -21.5, 23.8)

    # Plot only selected ecoregions
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    gdf.boundary.plot(
        ax=ax, 
        color='lightgrey', 
        linewidth=0.1
        )
    critical_gdf.plot(
        ax=ax, 
        color='red', 
        alpha=0.6,
        edgecolor='black', 
        linewidth=0.2
        )
    add_features(ax, extent=extent)
    # ax.set_title("Ecoregions with Combined Exposure in All PCA Axes")
    # ax.axis("off")

    # Save the figure
    plt.savefig("outputs/figures/critical_ecoregions_all_combinations.png", dpi=300)
    plt.show()

    # Save and display
    # merged_df.to_csv("outputs/csv/ecoregions_combined_effects_sorted.csv", index=False)
    # print(pivot_df)

   # Save the pivot table as a new CSV file

if __name__ == "__main__":
    run()