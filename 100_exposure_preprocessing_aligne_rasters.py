import xarray as xr
import rioxarray as rxr
import glob
import os

def run():
    """
    Align all rasters in the raster_folder and save them in the output_folder.

    The rasters are aligned using the outer join method, which ensures that all
    pixels are represented in the output rasters. The aligned rasters are saved
    with the same name as the original rasters, but with 'aligned_' prepended to
    the filename.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Define input/output folders
    raster_folder = "./inputs/03_rasters/correlations/"
    # raster_folder = "./inputs/03_rasters/biogeography/correlations_before_after/before/"
    # raster_folder = "./inputs/03_rasters/biogeography/correlations_before_after/after/"
    output_folder = "./outputs/raster/"
    # output_folder = "./outputs/raster/correlations_before_after/"
    os.makedirs(output_folder, exist_ok=True) # Ensure output folder exists

    # Find all .tif files in the folder
    raster_files = glob.glob(os.path.join(raster_folder, "*.tif"))

    # Create a dictionary with all the names and the corresponding rasters
    rasters = {os.path.basename(file): rxr.open_rasterio(file) for file in raster_files}

    # Align all rasters
    aligned_rasters = xr.align(*rasters.values(), join='outer') # align

    # Save each aligned raster separately
    for (filename, original_raster), aligned_raster in zip(rasters.items(), aligned_rasters):
        # Construct the output path
        # output_path = os.path.join(output_folder, f"aligned_{filename}")
        output_path = os.path.join(output_folder, f"aligned_{filename}")

        # Save the aligned raster
        aligned_raster.rio.to_raster(output_path)

        # Print a message to the user
        print(f"✅ Saved: {output_path}")

    # Print a final message
    print("All rasters aligned and saved.")

if __name__ == "__main__":  
    run()