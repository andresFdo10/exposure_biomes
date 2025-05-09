import rioxarray as rxr
import os
from rasterio.enums import Resampling
import glob

# Set folders
# raster_folder = "./inputs/03_rasters/biogeography/correlations_before_after/after/"
raster_folder = "./inputs/03_rasters/biogeography/correlations_before_after/long_ts/"
output_folder = "./outputs/raster/correlations_before_after/"
os.makedirs(output_folder, exist_ok=True)

# Load reference raster
# ref_raster_path = "inputs/03_rasters/biogeography/correlations_before_after/after/tmmx_nino_after.tif"
ref_raster_path = "inputs/03_rasters/biogeography/correlations_before_after/long_ts/tmmx_nino_60yrs.tif"
ref = rxr.open_rasterio(ref_raster_path, masked=True)

# Align all rasters to reference
for path in glob.glob(os.path.join(raster_folder, "*.tif")):
    raster = rxr.open_rasterio(path, masked=True)

    # Reproject and align to reference
    aligned = raster.rio.reproject_match(ref, resampling=Resampling.bilinear)

    # Save
    out_path = os.path.join(output_folder, f"aligned2_{os.path.basename(path)}")
    aligned.rio.to_raster(out_path)
    print(f"✅ Saved aligned raster: {out_path}")
