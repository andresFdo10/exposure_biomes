
# exposure_biomes

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/) 
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-GEE-green?logo=google)](https://earthengine.google.com/) 
[![Research Project](https://img.shields.io/badge/Project-Research-informational)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

> **exposure_biomes** is a research project focused on quantifying ecosystem exposure to climatic variability. Using the Entropy Weight Method (EWM), the project integrates correlations between the Oceanic Niño Index (ONI) and TerraClimate environmental variables to build an exposure index for both El Niño and La Niña events.  
> The workflow combines Google Earth Engine processing, local raster analysis, PCA, and spatial mapping of Neotropical ecoregions.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Installation and Requirements](#installation-and-requirements)
- [Project Structure](#project-structure)
- [Scripts Overview](#scripts-overview)
- [Outputs](#outputs)
- [Notes](#notes)

---

## About the Project

This project calculates an **exposure index** based on the strength of correlations between the ONI Index and TerraClimate variables. Analyses are performed separately for **El Niño** and **La Niña** phases to highlight climatic vulnerability across Neotropical ecosystems.

---

## Installation and Requirements

To run the Python scripts locally, you need the following packages:

```bash
pip install pandas geopandas numpy scikit-learn matplotlib rasterio
```

Other requirements:
- Access to [Google Earth Engine](https://earthengine.google.com/) for running `.js` scripts.
- Python 3.9 or higher recommended.

---

## Project Structure

- **Data Processing (Google Earth Engine)**:  
  Filtering and correlating TerraClimate variables based on ONI dates.
  
- **Preprocessing (Python)**:  
  Raster alignment, dataset merging, and exposure index computation.

- **Analysis**:  
  Entropy Weight Method calculations, PCA dimensionality reduction, and clustering of ecoregions.

- **Visualization and Output**:  
  Generation of maps, plots, and summary tables for critical regions.

---

## Scripts Overview

<details>
<summary>Click to expand</summary>

| Script | Description |
|:---|:---|
| **001_oni_terraclimate_correlations.js** | Filters TerraClimate variables for El Niño and La Niña periods using ONI Index. |
| **002_zonal_statistic_gee.js** | Extracts zonal statistics from selected variables. |
| **100_exposure_preprocessing_aligne_rasters.py** | Aligns raster datasets based on ONI–TerraClimate correlations. |
| **110_exposure_index_EWM.py** | Calculates the exposure index using the Entropy Weight Method. |
| **120_exposure_plot.py** | Generates a pixel-level plot of the exposure index. |
| **200_pre-processing_merge_datasets.py** | Merges datasets into a single GeoPackage. |
| **210_cumulative_exposure_plot.py** | Creates spatial distribution maps of environmental variables (Figure 1). |
| **310_pca_save_scores.py** | Performs PCA and saves transformed scores. |
| **330_pca_ecoregions_combined_effects.py** | Identifies ecoregions influenced by PC combinations. |
| **400_Figure2_ecoregions_clusters.py** | Generates clustered maps of ecoregions (Figure 2). |
| **500_Table1_summary_critical_ecoregions.py** | Summarizes critical ecoregions in a CSV table. |
| **502_TerraClimate_ONI_correlations_Figures_S1-S2.py** | Generates Figures S1 and S2: maps of TerraClimate–ONI correlation impacts. |

</details>

---

## Outputs

- Raster maps of the exposure index.
- PCA-based spatial distribution of climate drivers.
- Cluster maps of Neotropical ecoregions based on exposure patterns.
- Summary tables for critical ecoregions with environmental impact indicators.

---

## Notes

- Scripts prefixed with **0xx** run in **Google Earth Engine**.
- Scripts **1xx–5xx** are **Python-based** for local analysis and visualization.
- Outputs are ready for publication, figures, or further research.

---
