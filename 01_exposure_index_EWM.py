import os
from functools import reduce
import matplotlib.pyplot as plt
import rioxarray as rxr
import xarray as xr
import numpy as np

def exposure_signal(var, enso):
    """
    Calculate the exposure signal based on the input variable.
    The input variable corresponds to the correlation of each variable with
    the ONI Index.

    Parameters:
    var (str): The variable name to calculate the exposure signal for.
    enso (str): The ENSO phase ("elnino" or "lanina").

    Returns:
    xarray.DataArray: The exposure signal calculated based on the input variable.
    """

    # Build the path to the input raster
    path = f"./outputs/rasters/aligned_{var}_{enso}_60yrs.tif"

    # Check if the file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Open the raster and extract the significant correlation values
        data = rxr.open_rasterio(path)
        correlation = data[0]
        p_value = data[1]

        # Filter based on p-value significance
        significant_correlation = correlation.where(p_value < 0.05)

        # Check if there are any significant correlations
        if significant_correlation.isnull().all():
            raise ValueError("No significant correlations found for {var} during {enso}")

        # Set the exposure based on correlation values
        filter_correlation = xr.where(
            (var in ["tmmx", "tmmn", "vpd"]) & (correlation > 0)
            | (var not in ["tmmx", "tmmn", "vpd"]) & (correlation < 0),
            significant_correlation,
            0,
        )

        return filter_correlation

    except Exception as e:
        raise ValueError(
            f"An error occurred while processing {var} during {enso}: {e}"
        )
        return None


def normalize_variable(variable):
    """
    Normalize a variable by dividing each pixel's absolute value 
    by the total sum of absolute values.
    """
    total_sum = np.abs(variable).sum(skipna=True)
    if total_sum == 0:
        raise ValueError("No significant correlations found for normalization.")
    normalized = np.abs(variable) / total_sum
    return normalized


def calculate_entropy(normalized_variable):
    """
    Calculate the entropy of a variable
    """
    # Number of significant pixels
    n = normalized_variable.count().item()

    # Scaling constant
    k = 1 / np.log(n) if n > 1 else 0
    
    # Calculate the entropy
    entropy = -k * (normalized_variable * np.log(normalized_variable)).sum(skipna=True).item()

    return entropy


def calculate_weights(entropy_values):
    """
    Calculate weights for each variable based on their entropy values.
    Parameters:
    - entropy_values: dict, mapping variable names to their entropy values
    Returns:
    - weights: dict, mapping variable names to their calculated weights
    """
    # Calculate diversification for each variable
    diversification = {var: 1 - entropy for var, entropy in entropy_values.items()}
    
    # Sum of diversification values
    total_diversification = sum(diversification.values())
    
    # Calculate weights
    weights = {var: div / total_diversification for var, div in diversification.items()}
    return weights


def construct_exposure_index(normalized_variables, weights):
    """
    Construct the exposure index by combining normalized variables with weights.

    Parameters:
    - normalized_variables: dict, mapping variable names to normalized xarray.DataArray
    - weights: dict, mapping variable names to their calculated weights

    Returns:
    - exposure_index: xarray.DataArray, the constructed exposure index
    """
    import xarray as xr

    # Use one of the normalized variables to initialize the exposure index with zeros
    reference_variable = next(iter(normalized_variables.values()))
    exposure_index = xr.zeros_like(reference_variable)

    # Iterate through variables and combine their contributions
    for var, norm_data in normalized_variables.items():
        weighted_contribution = weights[var] * norm_data
        exposure_index += weighted_contribution

    return exposure_index


def run(enso):
    """
    Calculate and visualize the exposure index for a given region and ENSO phase.

    This function processes exposure signals for specified climate variables,
    normalizes them, calculates entropy, determines weights, and constructs a 
    vulnerability index. The resulting index is visualized and saved as a GeoTIFF.

    Parameters:
    region (str): The region of interest, either "coffea" or "theobroma".
    enso (str): The ENSO phase ("nino" or "nina").

    Returns:
    None
    """

    # Correlation variable to be loaded
    variables = ["tmmx", "tmmn", "pr", "pdsi"]

    # Step 1: Filter correlations and normalize
    filtered_signals = {var: exposure_signal(var, enso) for var in variables}
    normalized_signals = {var: normalize_variable(filtered_signals[var]) for var in filtered_signals}


    # Step 2: Calculate entropy for each variable
    entropy_values = {var: calculate_entropy(normalized_signals[var]) for var in normalized_signals}

    # Print entropy values
    print("Entropy Values:")
    for var, entropy in entropy_values.items():
        print(f"{var}: {entropy:.4f}")

    # Step 3: Determine weights
    weights = calculate_weights(entropy_values)

    # Print weights
    print("Weights:")
    for var, weight in weights.items():
        print(f"{var}: {weight:.4f}")

    # Step 4: Construct exposure index
    exposure_index = construct_exposure_index(normalized_signals, weights)
    exposure_index = exposure_index * 1e6

    # Scaling index
    exposure_index = (exposure_index - exposure_index.min())/(exposure_index.max() - exposure_index.min())

    exposure_index.plot()
    plt.ticklabel_format(style='plain', axis='both', useOffset=False) 
    plt.show()

    # Save the result to a GeoTIFF
    exposure_index.rio.to_raster(f"./outputs/rasters/exposure_EWM_{enso}.tif")

if __name__ == "__main__":
    run(enso="nina")