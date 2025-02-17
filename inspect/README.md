# Inspect Primary Cutouts

Aim is to inspect cutout to identify weaknesses, inaccuracies, and imblances. Inspection includes creating summary statistics csv and plots, and a sample of images for visual inspection. 

1. Prep data
    - check for cutout csvs
    - organize and clean
    - compile season csv data and save
2. Calculate stats
    - calculate counts
    - plot
    - compile stats and save
    - TODO: stats for cutout properties
3. Create inspection sample
    - check for data
    - create stratified sample of cropouts
    - create manual inspection form
    - compile sample and form, and save

## Prep data

### Check for data

- check for to makes sure all csvs exists for all processed batches. Give warning if not.
- get processed csvs
- check that all processed csvs exists

### Compile csv data

read csvs, convert to pandas dataframses, and concat all dataframes

### Organize and clean

Organize dataframe:
- include only `is_primary` cutouts
- add `state_id` feature
- use `common_name` as class identifier
- create `bordering` feature that describes neighboring plant-species groups in the potting area
- create `temp_cropout_path` feature for crops outs (not cutouts) in dataframe

## Descriptive stats

Creates CSVs and plot figures for assessing the overall image count and species distribution across locations and batches. Future plans should include looking at cutout properties. 

### Calculate stats

Count:
1. total images
2. total cutouts (primary and non-primary)
3. total images and cutouts per batch
4. species by location

## Create inspection sample

Inspection sample to be used for manual inspecting cropouts and verifying species class names.

### Check for data

Check if data exists. Download if it does not.

### Stratified sampling

Random sample of 20 cropouts is selected for each species in each batch. We applies pandas [`groupby`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) and [`sample`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html). Random sampling is seeded for reproducibility.

### Create inspection form

A csv that manual cropout inspectors can use to verify class accuracy for each output image


This project is designed to inspect primary cutouts to identify weaknesses, inaccuracies, and imbalances. The inspection process includes generating summary statistics CSVs and plots, along with sampling images for visual inspection.

## Project Structure
```
.
├─README.md
├─inspect_cutouts.py
└─inspect_utils
  ├─cutout_descriptive_stats.py
  ├─data_classes.py
  ├─inspect_utils.py
  └─viz.py
```

## Workflow

### 1. Preparing Data
- **Check for cutout CSVs**: Ensure that all required CSV files exist for all processed batches and issue warnings if any are missing.
- **Organize and clean**: Process and clean data to include only primary cutouts, add state identifiers, and create features such as `bordering` and `temp_cropout_path`.
- **Compile season CSV data**: Convert CSVs to pandas DataFrames and concatenate them into a single dataset.

### 2. Calculating Statistics
- **Generate counts**:
  1. Total images
  2. Total cutouts (primary and non-primary)
  3. Images and cutouts per batch
  4. Species distribution by location
- **Generate plots**: Create visualizations to understand dataset imbalances and distributions.
- **Compile statistics**: Aggregate and save statistical outputs.
- **TODO**: Expand statistics to include cutout properties.

### 3. Creating Inspection Samples
- **Check for data**: Ensure that required data is available for processing.
- **Generate stratified samples**: Use `pandas.DataFrame.groupby` and `sample` functions to generate a reproducible random sample of 20 cropouts per species per batch.
- **Create inspection forms**: Generate CSVs for manual inspectors to verify class labels for sampled cropouts.

## Code Details

### `inspect_cutouts.py`
The main entry point for running cutout inspections. It compiles the data, calculates statistics, and facilitates manual inspections.

### `inspect_utils/cutout_descriptive_stats.py`
Generates descriptive statistics and visualizations for cutout distributions.

### `inspect_utils/data_classes.py`
Defines data structures for handling cutout inspection samples, including batch and species grouping.

### `inspect_utils/inspect_utils.py`
Includes utility functions for reading CSVs, processing metadata, performing manual inspections, and handling GUI interactions.

### `inspect_utils/viz.py`
Contains visualization utilities, including functions for generating confusion matrices and other diagnostic plots.

## Running the Inspection
To inspect primary cutouts, execute the following:
```sh
python inspect_cutouts.py --config-path <path_to_config>
```
Ensure that the necessary dependencies, including `pandas`, `matplotlib`, `seaborn`, and `opencv-python`, are installed.

## Future Improvements
- Expand statistical analysis to include cutout property distributions.
- Automate data retrieval from storage.
- Improve sampling strategies for enhanced representation.
- Develop interactive visual dashboards for inspection results.

