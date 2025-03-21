import pandas as pd

def run():

    # Load the computed influence scores
    influence_scores_df = pd.read_csv("outputs/csv/ecoregion_variable_influence_scores.csv")

    # Define the top 10% threshold for each variable
    thresholds = influence_scores_df.iloc[:, 1:].quantile(0.75)  # Exclude 'ECO_ID'

    # Identify ecoregions in the top 10% for each variable
    top_affected_ecoregions = {}
    for variable in thresholds.index:
        top_affected_ecoregions[variable] = influence_scores_df[influence_scores_df[variable] >= thresholds[variable]]

    # Combine results into a single DataFrame
    combined_effects_list = []
    for variable, df in top_affected_ecoregions.items():
        for _, row in df.iterrows():
            combined_effects_list.append({"ECO_ID": row["ECO_ID"], "Most Influenced By": variable})

    combined_effects_df = pd.DataFrame(combined_effects_list).drop_duplicates()

    # Identify ecoregions experiencing multiple effects
    multi_effect_ecoregions = combined_effects_df["ECO_ID"].value_counts()
    combined_ecoregions = multi_effect_ecoregions[multi_effect_ecoregions > 4].index

    # Create a final table highlighting ecoregions with combined effects
    final_combined_df = combined_effects_df[combined_effects_df["ECO_ID"].isin(combined_ecoregions)]

    # Save results
    combined_effects_df.to_csv("outputs/csv/top_10_percent_ecoregions_per_variable.csv", index=False)
    final_combined_df.to_csv("outputs/csv/ecoregions_with_combined_effects_2.csv", index=False)
    print(final_combined_df.drop_duplicates(subset=["ECO_ID"]))
    # print(combined_effects_df)

    print("Filtering complete. Results saved in 'outputs/csv' folder.")

if __name__ == "__main__":
    run()