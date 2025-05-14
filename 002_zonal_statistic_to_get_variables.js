var loss_by_fires = ee.Image("projects/ee-dcriticalagroforestry/assets/Fires"),
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017"),
    hansen_gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11"),
    primary_forest = ee.ImageCollection("UMD/GLAD/PRIMARY_HUMID_TROPICAL_FORESTS/v1");

// Select tree cover and loss bands
var treecover_2000 = hansen_gfc.select(['treecover2000']);
print(treecover_2000);
var lossImage = hansen_gfc.select(['loss']);
print(lossImage);
var lossYear = hansen_gfc.select(['lossyear']); // Loss year image
var primaryForest = primary_forest.first();
// Load the Ecoregions dataset

// Filter only Neotropical ecoregions
var neotropicalEcoregions = ecoregions.filter(ee.Filter.eq('REALM', 'Neotropic'));
Map.addLayer(neotropicalEcoregions);

// Define pixel area in hectares (1 ha = 10,000 m²)
var pixelAreaHa = ee.Image.pixelArea().divide(10000);

// Define forest cover threshold (change if needed)
var treecover50 = treecover_2000.gte(50).updateMask(treecover_2000.gte(50));
// Compute total forest area in 2000 using the treecover 50%
// Var 1
var forestArea50 = treecover50.multiply(pixelAreaHa);

// Define Primary forest cover by threshold
var primaryForest50 = primaryForest.updateMask(treecover50);
// Var 2
var primaryForest50_Area = primaryForest.multiply(pixelAreaHa);

// Compute total forest loss (2001-2023)
var totalLoss50 = lossImage.updateMask(treecover50);
// Var 3
var totalLoss50_Area = totalLoss50.multiply(totalLoss50);

var totalPrimaryLoss50 = totalLoss50.updateMask(primaryForest50);
// Var 4
var totalPrimaryLossArea50 = totalPrimaryLoss50.multiply(pixelAreaHa);

// Compute total forest loss by fires according to threshold
var totalLoss_fires = loss_by_fires.updateMask(totalLoss50);
// Var 5
var totalLoss_fires50 = totalLoss_fires.multiply(pixelAreaHa);

// Compute total forest loss by fires and the areas
var totalPrimaryLoss_fires50 = loss_by_fires.updateMask(totalPrimaryLoss50);
//var totalPrimaryLoss_by_fires50 = totalLoss_by_fires.updateMask(pforest);
// Var 6
var PrimaryLoss_firesAreas50 = totalPrimaryLoss_fires50.multiply(pixelAreaHa);

// Function to compute zonal statistics including total forest loss and velocity per year
var computeZonalStats = function(region) {
  var regionName = region.get('ECO_NAME'); // Adjust field name if necessary
  
    // Compute total initial forested area (ha)
  var stats1 = forestArea50.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: region.geometry(),
    scale: 30,
    maxPixels: 1e13
  });

  var ForestArea50 = ee.Number(stats1.get('treecover2000'));

  
  // Compute total initial forested area (ha)
  var stats2 = primaryForest50_Area.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: region.geometry(),
    scale: 30,
    maxPixels: 1e13
  });

  var PrimaryForestArea = ee.Number(stats2.get('Primary_HT_forests'));
  
  // Compute total initial forested area (ha)
  var stats3 = totalLoss50_Area.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: region.geometry(),
    scale: 30,
    maxPixels: 1e13
  });

  var ForestLoss50Area = ee.Number(stats3.get('loss'));
  
  // Compute total initial forested area (ha)
  var stats4 = totalPrimaryLossArea50 .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: region.geometry(),
    scale: 30,
    maxPixels: 1e13
  });

  var PrimaryForestLoss50Area = ee.Number(stats4.get('loss'));
  
  // Compute total initial forested area (ha)
  var stats5 = totalLoss_fires50.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: region.geometry(),
    scale: 30,
    maxPixels: 1e13
  });

  var ForestLossFires50Area = ee.Number(stats5.get('b1'));
  
  // Compute total forest loss (2001-2023)
  var Stats6 = PrimaryLoss_firesAreas50.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: region.geometry(),
    scale: 30,
    maxPixels: 1e13
  });

  var PrimaryLossFires50Area = ee.Number(Stats6.get('b1'));
  
  // Proportion of loss-induced by fires
    var PrimaryLoss_FiresPerc = PrimaryLossFires50Area.divide(PrimaryForestLoss50Area).multiply(100);
    
  // Primary forest rate
  var years = ee.Number(23);
  var primary_loss_rate = PrimaryForestLoss50Area.divide(PrimaryForestArea.multiply(years)).multiply(100);

  // Set the attributes for the region
  return region.set({
    'Forest50':ForestArea50,
    'PrimaryForest50': PrimaryForestArea,
    'Forest_Loss50': ForestLoss50Area,
    'PrimaryForest_Loss50': PrimaryForestLoss50Area,
    'PrimaryLoss_Rate50': primary_loss_rate,
    'ForestLoss_Fires50': ForestLossFires50Area,
    'PrimaryForestLoss_Fires50':PrimaryLossFires50Area,
    'proportion_fire-induced_Primaryloss': PrimaryLoss_FiresPerc,
    'ECO_NAME': regionName
  });
};

// Apply function to all ecoregions
var zonalStats = neotropicalEcoregions.map(computeZonalStats);

// Print the results
// print("Total Forest Loss Rate (%/year)", zonalStats);

// Export results as a CSV file for further analysis
Export.table.toDrive({
  collection: zonalStats,
  description: 'ZonalStatistics_Ecoregions_Complete',
  fileFormat: 'CSV'
});
