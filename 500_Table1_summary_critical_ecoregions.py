import pandas as pd
import geopandas as gpd
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

    # Save and display
    merged_df.to_csv("outputs/csv/ecoregions_combined_effects_sorted.csv", index=False)
    # print(pivot_df)

   # Save the pivot table as a new CSV file

if __name__ == "__main__":
    run()