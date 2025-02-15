# README: Metadata Cleaning and Validation Pipeline

## Project Structure
```
.
├─inspect_validation.py
├─update_and_move.py
├─validate_utils.py
└─validate_weed_classifier.py
```

## Overview
This repository contains scripts for processing, validating, and organizing metadata and image data within an agricultural image processing pipeline. The pipeline ensures data consistency, correctness, and efficient storage management.

## Scripts
### 1. `inspect_validation.py`
This script facilitates manual validation of images by displaying images with associated bounding boxes, masks, and cutouts. Users can classify images using keyboard shortcuts.

- **Features**:
  - Loads validation images and displays them with annotations.
  - Allows manual classification: pass, fail, or review.
  - Saves validation results to a CSV file.
  - Ensures image scaling for better visualization.
  
- **Usage**:
  ```sh
  python inspect_validation.py
  ```
  
- **Key Actions**:
  - `a` - Mark image as passed.
  - `s` - Mark image as failed.
  - `t` - Mark image for further review.
  - `q` - Quit the validation process.

### 2. `update_and_move.py`
Once validation is complete, this script updates metadata and transfers files to long-term storage.

- **Features**:
  - Updates metadata JSON files by setting the `validated` key to `True`.
  - Copies validated images and metadata to a long-term storage (LTS) location.
  - Ensures files are correctly copied before deleting local versions.
  
- **Usage**:
  ```sh
  python update_and_move.py
  ```
  
- **Actions**:
  - Updates metadata validation status.
  - Copies full-sized images and cutouts to LTS.
  - Confirms file consistency before removing local versions.

### 3. `validate_utils.py`
This script contains helper functions for metadata validation, visualization, and analysis.

- **Features**:
  - Reads and processes metadata.
  - Generates and plots bounding boxes, masks, and cutout overlays.
  - Provides sampling strategies for validation subsets.
  - Supports data aggregation and filtering by species, area, solidity, etc.

- **Usage**:
  ```python
  from validate_utils import batch_df, get_bboxes_validation_images
  ```
  
  Example function call:
  ```python
  batch_metadata = batch_df(batch_id="TX_2024-07-07", cutout_dir="/data/cutouts", batch_dir="/data/batches")
  ```

### 4. `validate_weed_classifier.py`
This script reviews plant detection results and visually annotates bounding boxes for high-confidence weed classifications.

- **Features**:
  - Reads classifier results from CSV files.
  - Identifies and marks images containing `non_target_weed` classifications with high confidence.
  - Generates annotated images with bounding boxes and cropped insets.
  
- **Usage**:
  ```sh
  python validate_weed_classifier.py
  ```

## Requirements
Ensure the following dependencies are installed:
```sh
pip install pandas numpy opencv-python matplotlib seaborn hydra-core tqdm
```

## Configuration
The scripts use Hydra for configuration management. Update `config.yaml` in the `conf/` directory to specify paths and settings.

## Workflow
1. **Validate Images** (`inspect_validation.py`) - Manually classify validation images.
2. **Update Metadata & Transfer Files** (`update_and_move.py`) - Move validated images to LTS.
3. **Metadata Analysis** (`validate_utils.py`) - Perform batch processing and visualization.
4. **Review Weed Classification** (`validate_weed_classifier.py`) - Check weed classifier results.

