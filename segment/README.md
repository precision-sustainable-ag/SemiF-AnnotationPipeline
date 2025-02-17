# Script Documentation

## Overview
This directory contains scripts related to species assignment, structure-from-motion automation, plant localization, bounding box management, weed classification, and vegetation segmentation in agricultural image processing.

# Directory Structure
```
.
├─assign_species.py
├─auto_sfm.py
├─bbox
├─classify_nontarget_weeds.py
├─convert.py
├─localize_plants.py
├─merge_overlapping_bboxes.py
├─remap_labels.py
└─segment_vegetation.py
```

## Script Descriptions

### 1. `assign_species.py`
- **Purpose:** Assign species labels to bounding boxes by comparing their centroid points with shapefile polygons.
- **Key Features:**
  - Uses Geopandas to process shapefile data.
  - Checks if the bounding box centroid falls inside a polygon.
  - If a match is not found, assigns the nearest polygon within 1m.
  - Saves updated metadata after assignment.

### 2. `auto_sfm.py`
- **Purpose:** Automates the Structure from Motion (SfM) pipeline for image processing.
- **Key Features:**
  - Saves AutoSfM configuration in YAML.
  - Copies developed images and masks to structured locations.
  - Runs AutoSfM in a Docker container, supporting GPU processing.
  - Moves exported outputs to batch directory and cleans temporary storage.

### 3. `classify_nontarget_weeds.py`
- **Purpose:** Classifies weeds as target or non-target species using a YOLO model.
- **Key Features:**
  - Reads CSV detections and their associated images.
  - Crops bounding box regions and classifies using a pretrained YOLO model.
  - Outputs results with classifier predictions.
  - Configurable per-state and season classification.

### 4. `convert.py`
- **Purpose:** Processes and cleans metadata files for cutout and full-size images.
- **Key Features:**
  - Converts JSON metadata into structured CSVs.
  - Uses defined rules to clean and organize metadata.
  - Supports both full-size images and cutouts.
  - Saves cleaned metadata in structured directories.

### 5. `localize_plants.py`
- **Purpose:** Merges detection results from multiple CSVs to create a single detection output file.
- **Key Features:**
  - Reads CSVs from plant detection results.
  - Filters and validates bounding box coordinates.
  - Combines results and saves into a single detection file.

### 6. `merge_overlapping_bboxes.py`
- **Purpose:** Merges overlapping or duplicate bounding boxes while preserving classification labels.
- **Key Features:**
  - Uses IoU-based merging to detect duplicate bounding boxes.
  - Retains class labels (e.g., colorchecker, plant).
  - Outputs refined bounding box lists to CSV.

### 7. `remap_labels.py`
- **Purpose:** Maps bounding boxes from local image coordinates to global coordinates using SfM-generated transformations.
- **Key Features:**
  - Reads YOLO annotations and SfM reference files.
  - Applies global transformation to bounding boxes.
  - Uses non-max suppression to remove duplicate boxes.
  - Saves refined bounding box metadata.

### 8. `segment_vegetation.py`
- **Purpose:** Segments vegetation in images using deep learning-based and clustering-based segmentation methods.
- **Key Features:**
  - Uses a segmentation model for mask prediction.
  - Generates semantic and instance segmentation masks.
  - Saves segmented masks, metadata, and associated cutouts.
  - Supports both sequential and parallel execution for efficiency.

## Usage
Each script is designed to be executed within a structured processing pipeline, typically using `omegaconf.DictConfig` for configuration. The pipeline can be run sequentially or in parallel, depending on system resources and processing needs.

## Dependencies
These scripts rely on various external libraries, including:
- `geopandas` for spatial data processing.
- `opencv` for image manipulation.
- `YOLO` for object detection.
- `networkx` for graph-based bounding box merging.
- `torch` for deep learning-based segmentation.
- `shapely` for geometric operations.

Ensure that the necessary dependencies are installed before executing the scripts.

## Notes
- The pipeline expects a structured dataset format with batch directories, metadata, and processed outputs.
- Configuration files (YAML) should be properly set before execution.
- Logging is enabled for debugging and tracking progress.
