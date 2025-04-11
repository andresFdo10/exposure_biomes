# exposure_biomes
This project calculates the exposure index using the Entropy Weight Method from correlations between ONI index and Terraclimate variables. This index is generated both El Niño and La Niña Events. 


001_oni_terraclimate_correlations.js: Filter imageCollection terraclimate variables for El Niño and La Niña dates acordding to el ONI Index which is loaded as a table.
002_zonal_statistic_gee.js: Extract the information of the variables used in this analysis.
100_exposure_preprocessing_aligne_rasters.py: Aligne the rasters corresponding to the correlations beteween the terraclimate variables and the ONI Index.
110_exposure_index_EWM.py: Calculate the exposure index using the Entropy Weight Method for the correlations between Terraclimate variables and the ONI Index.
120_exposure_plot.py: Show the plot of the exposure index pixel by pixel.
200_pre-processing_merge_datasets.py: Load the ecoregions from a geopackage file. New fields are added to this dataset, they corresponds to the  
