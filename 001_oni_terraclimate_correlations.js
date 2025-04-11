/*************************        SETTINGS & REGIONS       *************************/

// Load the ONI index table (uploaded to GEE Assets)
var oni_index = ee.FeatureCollection("projects/ee-dcriticalagroforestry/assets/oni_index");
// Define a point for time series visualization
var roi = /* color: #d63000 */ee.Geometry.Point(
        [-74.72041018490758, 3.8285357170778074]
    );

// Neotropic region
var Region = 
    ee.Geometry.Polygon(
        [[[-116.22014877686846, 23.5],
          [-116.22014877686846, -66.5],
          [-31.493586276868474, -66.5],
          [-31.493586276868474, 23.5]]], null, false);

Map.setCenter(-67.06, -0.87, 5);

/***************************      ONI Index    *******************************/
/**
 * Prepare an ONI image using TerraClimate's 'pdsi' band.
 * This is a placeholder image used to carry ONI values across time.
 *
 * NOTE: To use a different climate variable (e.g., 'tmmx', 'ppt'), 
 * replace all instances of 'pdsi' in this script using Ctrl+H 
 * (search and replace).
 */
var target_dataset = ee.Image('IDAHO_EPSCOR/TERRACLIMATE/195812')
    .select('pdsi')
    .rename(['oni']);

/**
 * Format the ONI 'ANOM' column to ensure numeric values.
 * @param {ee.Feature} feature - A feature from the ONI table
 * @returns {ee.Feature} - The same feature with 'ANOM' cast to Number
 */
var formatAnom = function(feature) {
  var formattedAnom = ee.Number.parse(feature.get('ANOM')); // Convert string to number
  return feature.set('ANOM', formattedAnom);
};

// Apply formatting and sort the ONI table
oni_index = oni_index.map(formatAnom);

oni_index = oni_index.sort('SEAS2').sort('YR');

// Extract ordered ONI anomaly values
var anomList = oni_index.aggregate_array('ANOM');

var sequence = ee.List.sequence(0, anomList.length().subtract(1), 1);

// Create list of dates to align ONI values in time
var date_start = ee.Date('1950-01-01');
var date_end = ee.Date('2024-08-31');
var n_months = date_end.difference(date_start,'month').round();
var sequence_dates = ee.List.sequence(0, n_months.subtract(1), 1);
var make_datelist = function(n) {
  return date_start.advance(n,'month');
};

var dates = sequence_dates.map(make_datelist);

/**
 * Create pixel-by-pixel ONI ImageCollection using anomaly values.
 * @param {ee.Number} i - Index of the ONI value
 * @returns {ee.Image} - Image with ONI value as constant for all pixels
 */
var ONI = function(i){
  return target_dataset
    .divide(target_dataset)
    .multiply(ee.Number(anomList.get(i)))
    .set({'system:time_start': ee.Date(dates.get(i)).millis()})};
    
var enso_ONI = sequence.map(ONI);
var enso_ONI = ee.ImageCollection.fromImages(enso_ONI);

/****************************      DATASET        ****************************/
var dataset = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
                  .filter(ee.Filter.date('1958-12-01', '2023-12-31'))
                  .filterBounds(Region)
                  .select('pdsi');

/*************************       ANOMALIES       *****************************/
// Define time windows and pivot years for anomaly analysis
var start1 = ee.List(
  [
    '1961-01-01', '1961-01-01', '1961-01-01', '1961-01-01', 
    '1966-01-01', '1971-01-01', '1976-01-01', '1981-01-01',
    '1986-01-01', '1991-01-01', '1991-01-01', '1991-01-01',
    '2021-01-01'
  ]
); // Start of 30-year means

var end1 = ee.List(
  [
    '1990-12-31', '1990-12-31', '1990-12-31', '1990-12-31',
    '1995-12-31', '2000-12-31', '2005-12-31', '2010-12-31', 
    '2015-12-31', '2020-12-31', '2023-12-31', '2023-12-31',
    '2023-12-31'
  ]
); // End of 30-year means

var year = ee.List(
  [1961, 1966, 1971, 1976, 1981, 1986, 1991, 1996, 2001, 2006, 2011, 2016, 2021]
);  // 5-year periods for anomalies


var sequence = ee.List.sequence(0, start1.length().subtract(1), 1);

/**
 * Computes anomalies based on 30-year mean and 5-year analysis periods.
 * @param {ee.Number} i - Index in time period list
 * @returns {ee.ImageCollection} - Anomaly images
 */
var anomaly = function(i) {
  var start = ee.Date(start1.get(i));
  var end = ee.Date(end1.get(i));
  var yearStart = ee.Date.fromYMD(year.get(i), 1, 1);
  var yearEnd = ee.Date.fromYMD(ee.Number(year.get(i)).add(4), 12, 31);

  var pp_yearmon1 = ee.List.sequence(1, 12).map(function(n) {
    var tclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
      .filterDate(start, end)
      .filter(ee.Filter.calendarRange(n, n, 'month'))
      .filterBounds(Region)
      .select('pdsi');
      
    // Calculate the mean for the selected month
    var mean = tclimate.reduce(ee.Reducer.mean());
    
    // Subtract the monthly mean from each image in the 5-year period to get anomalies
    return tclimate.filter(ee.Filter.date(yearStart, yearEnd)).map(function(image) {
      return image.subtract(mean).set('system:time_start', image.get('system:time_start'));
    });
  });

  // Merge all monthly anomaly collections into a single ImageCollection
  var icMerged1 = ee.ImageCollection(pp_yearmon1.get(0))
    .merge(ee.ImageCollection(pp_yearmon1.get(1)))
    .merge(ee.ImageCollection(pp_yearmon1.get(2)))
    .merge(ee.ImageCollection(pp_yearmon1.get(3)))
    .merge(ee.ImageCollection(pp_yearmon1.get(4)))
    .merge(ee.ImageCollection(pp_yearmon1.get(5)))
    .merge(ee.ImageCollection(pp_yearmon1.get(6)))
    .merge(ee.ImageCollection(pp_yearmon1.get(7)))
    .merge(ee.ImageCollection(pp_yearmon1.get(8)))
    .merge(ee.ImageCollection(pp_yearmon1.get(9)))
    .merge(ee.ImageCollection(pp_yearmon1.get(10)))
    .merge(ee.ImageCollection(pp_yearmon1.get(11)));

  return icMerged1;
};

var anomalies = sequence.map(anomaly);


// Warning: It is important to consider how many marge it is necessary to apply.
var anomalies = (ee.ImageCollection(anomalies.get(0)))
  .merge(ee.ImageCollection(anomalies.get(1)))
  .merge(ee.ImageCollection(anomalies.get(2)))
  .merge(ee.ImageCollection(anomalies.get(3)))
  .merge(ee.ImageCollection(anomalies.get(4)))
  .merge(ee.ImageCollection(anomalies.get(5)))
  .merge(ee.ImageCollection(anomalies.get(6)))
  .merge(ee.ImageCollection(anomalies.get(7)))
  .merge(ee.ImageCollection(anomalies.get(8)))
  .merge(ee.ImageCollection(anomalies.get(9)))
  .merge(ee.ImageCollection(anomalies.get(10)))
  .merge(ee.ImageCollection(anomalies.get(11)))
  .merge(ee.ImageCollection(anomalies.get(12)));


/** 
 * *************************       SEASONALITY      ***************************
**/
// Define 3-month moving average periods
var date_start = ee.Date('1960-12-01');
var date_end = ee.Date('2023-12-31');  // Extended end date to 2023
var n_months = date_end.difference(date_start, 'month').round();
var dates = ee.List.sequence(0, n_months.subtract(1), 1);
var make_datelist = function(n) {return date_start.advance(n, 'month')};
dates = dates.map(make_datelist);

// -----------------------------------------------------------------------------

var sequence2 = ee.List.sequence(0, dates.length().subtract(3), 1);
// print('sequence2', sequence2);

var season = function(i) {
  var accumulate = anomalies
    .filterDate(ee.Date(dates.get(i)), ee.Date(dates.get(i)).advance(3, 'month'))
    .select('pdsi');

  var mean = accumulate
    .reduce(ee.Reducer.mean())
    .set({
      'system:time_start': ee.Date(dates.get(i)).advance(1, 'month').millis()
    });
  return mean;
};

// ------------------------------------------------------------------------------

var first = sequence2.map(season);

// Convert the list of images into an image collection.
var anomalies2 = ee.ImageCollection.fromImages(first);


/*************************       CORRELATION      ****************************/
// Filter ONI and pdsi datasets to the same period and region
var enso_filter = enso_ONI
  .filterDate(ee.Date('1961-01-01'), ee.Date('2021-12-31'))
  .filterBounds(Region)
  .map(function(img){return img.rename('oni')});
print(ui.Chart.image.series(enso_filter.select('oni'), roi)
  .setChartType('LineChart')
  .setOptions({
    title: 'ONI',
    lineWidht:1
  }));


var tclimate_filter = anomalies2
  .filterDate(ee.Date('1961-01-01'), ee.Date('2021-12-31'))
  .filterBounds(Region);
  

/**
 * Adds the mean ONI value over the region to each image.
 * @param {ee.Image} image - Image with ONI band
 * @returns {ee.Image} - Image with added mean_oni property
 */
var calculateMean = function(image) {
  var mean = image.reduceRegion({
    reducer: ee.Reducer.mean(), // Use the mean reducer
    geometry: Region, // Define the region of interest
    scale: image.select(0).projection().nominalScale(), // Use the image's nominal scale
    maxPixels: 1e13 // Maximum number of pixels to process
  }).get('oni'); // Retrieve the mean value of the 'oni' band

  return image.set('mean_oni', mean);
};

// Map the function over the ImageCollection to add mean as a property
var imageCollectionWithMean = enso_filter.map(calculateMean);
  
// Split into El Niño and La Niña cases
var meanLessThanZero = imageCollectionWithMean.filter(ee.Filter.lt('mean_oni', -0.5));
var meanGreaterThanZero = imageCollectionWithMean.filter(ee.Filter.gt('mean_oni', 0.5));
  
// Optional: Visualize the filtered ImageCollection
// Map.addLayer(meanLessThanZero, {}, 'Filtered Images La Nina');
// Map.addLayer(meanGreaterThanZero, {}, 'Filtered Images El Nino');

// Extract the list of dates from the filtered primary ImageCollection
var filterNegativeValues = meanLessThanZero.aggregate_array('system:time_start');
var filterPositiveValues = meanGreaterThanZero.aggregate_array('system:time_start');


// Extract matching pdsi images for each ONI event
var filterByDate = function(date) {
  date = ee.Date(date);
  return tclimate_filter.filterDate(date, date.advance(1, 'day')).first();
};
  
var ElNino = filterPositiveValues.map(filterByDate);
var LaNina = filterNegativeValues.map(filterByDate);
  
var filteredByElNinoImageCollection = ee.ImageCollection(ElNino);
var filteredByLaNinaImageCollection = ee.ImageCollection(LaNina);


/**
 * Synchronizes pdsi and ONI El Nino images by timestamp.
 * @param {ee.ImageCollection} tcCollection - TerraClimate images
 * @param {ee.ImageCollection} oniCollection - ONI images
 * @returns {ee.ImageCollection} - Collection with merged bands
 */
var syncedCollection_nino = filteredByElNinoImageCollection.map(function(tcImage) {
  var correspondingONI = meanGreaterThanZero
    .filter(ee.Filter.date(tcImage.date()))
    .first(); // Get the first matching ONI image
  correspondingONI = ee.Image(correspondingONI).rename('oni');
  return tcImage.addBands(correspondingONI);
});


/**
 * Synchronizes pdsi and ONI La Nina images by timestamp.
 * @param {ee.ImageCollection} tcCollection - TerraClimate images
 * @param {ee.ImageCollection} oniCollection - ONI images
 * @returns {ee.ImageCollection} - Collection with merged bands
 */
var syncedCollection_nina = filteredByLaNinaImageCollection.map(function(tcImage) {
  var correspondingONI = meanLessThanZero
    .filter(ee.Filter.date(tcImage.date()))
    .first(); // Get the first matching ONI image
  correspondingONI = ee.Image(correspondingONI).rename('oni');
  return tcImage.addBands(correspondingONI);
});


// Display synchronized ImageCollection
print(ui.Chart.image.series(syncedCollection_nina.select(['pdsi_mean','oni']), roi)
  .setChartType('LineChart')
  .setOptions({
    title: 'pdsi and Nina',
    lineWidht:1
  }));

print(ui.Chart.image.series(syncedCollection_nino.select(['pdsi_mean','oni']), roi)
  .setChartType('LineChart')
  .setOptions({
    title: 'pdsi and Nino',
    lineWidht:1
  }));

// Compute Pearson correlation
var correlation_nino = syncedCollection_nino.reduce(ee.Reducer.pearsonsCorrelation());
var correlation_nina = syncedCollection_nina.reduce(ee.Reducer.pearsonsCorrelation());

/************************         EXPORT RESULTS        **********************/
// Export the image to an Earth Engine asset.
Export.image.toAsset({
  image: correlation_nino,
  description: 'pdsi_Pearson_nino_60years',
  assetId: 'pdsi_Pearson_nino_60years',
  //crs: projection.crs,
  scale: 4638,
  //crsTransform: projection.transform,
  region: Region,
  //pyramidingPolicy: {'.default': 'sample'}
});

// Export the image to an Earth Engine asset.
Export.image.toAsset({
  image: correlation_nina,
  description: 'pdsi_Pearson_nina_60years',
  assetId: 'pdsi_Pearson_nina_60years',
  //crs: projection.crs,
  scale: 4638,
  //crsTransform: projection.transform,
  region: Region,
  //pyramidingPolicy: {'.default': 'sample'}
});