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