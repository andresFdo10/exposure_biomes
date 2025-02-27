import xarray as xr
import rioxarray as rxr
import glob
import os

def run():

    # Define input/output folders
    raster_folder = "./inputs/raster/biogeography/"
    output_folder = "./outputs/raster/"
    os.makedirs(output_folder, exist_ok=True) # Ensure output folder exists

    raster_files = glob.glob(os.path.join(raster_folder, "*.tif")) # Get all the .tif files in the folder
    rasters = {os.path.basename(file): rxr.open_rasterio(file) for file in raster_files} # Dict with all the names

    # algin all rasters
    aligned_rasters = xr.align(*rasters.values(), join='outer') # align
    # Load all rasters
    # raster_files = glob.glob("./inputs/rasters/biogeography/*.tif")
    # rasters = [rxr.open_rasterio(file) for file in raster_files]

    # Check if they have tha same shape and coordinates
    # for i, r in enumerate(rasters):
    #     print(f"Raster {i}: shape= {r.shape}, crs= {r.crs}")

    # Align all rasters (using 'outer' to include all data, 'inner' to keep common areas only)

    # Stack them into a single dataset for easier processing (if needed)
    # stacked_rasters = xr.stack(aligned_rasters, dim='variable')

    # # Perform your operation (e.g., sum all rasters)
    # result = stacked_rasters.sum(dim='variable')

    # output path
    # output_folder = "./outputs/rasters/"

    # Save each aligned raster separately
    for (filename, original_raster), aligned_raster in zip(rasters.items(), aligned_rasters):
        output_path = os.path.join(output_folder, f"aligned_{filename}")
        aligned_raster.rio.to_raster(output_path)
        print(f"✅ Saved: {output_path}")

    print("All rasters aligned and saved.")

if __name__ == "__main__":  
    run()