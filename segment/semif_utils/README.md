# Documentation for Codebase

## Overview
This repository contains scripts for dataset handling, model training and inference, image segmentation, and utility functions. Below is an overview of the scripts and their functionalities.

---
## Folder Structure
```
.
├─ datasets.py          # Handles dataset structures and metadata
├─ model.py             # Defines machine learning models and inference
├─ segment_species.py   # Species segmentation methods
├─ segment_utils.py     # Utility functions for segmentation
└─ utils.py             # General utility functions
```

---
## Script Details

### 1. `datasets.py`
This script defines data classes and methods for handling image datasets, bounding boxes, and metadata.

**Main Components:**
- `BoxCoordinates`: Stores bounding box coordinates and provides scaling capabilities.
- `BBox`: Represents a bounding box with metadata, including area and centroid calculations.
- `BatchMetadata`: Stores batch-level metadata.
- `ImageMetadata`: Holds EXIF metadata for images.
- `ImageData` and `RemapImage`: Handle image data processing and metadata extraction.
- `Cutout`: Represents segmented cutouts from images with associated metadata.

---
### 2. `model.py`
This script contains machine learning model implementations for segmentation and inference.

**Main Components:**
- `MaskPredictor`: A class for handling mask prediction, including preprocessing and inference.
- `SegmentationModule`: Implements a PyTorch Lightning module for segmentation, supporting multiple architectures like Unet and DeepLabV3Plus.
- Training and validation logging, loss computation, and metric tracking.

---
### 3. `segment_species.py`
Handles species segmentation and processing using various computer vision techniques.

**Main Components:**
- `Segment`: Class for segmenting species using vegetation indices, clustering, and thresholding.
- Implements methods like:
  - `general_seg()`: General segmentation using clustering or thresholding.
  - `watershed()`: Applies watershed segmentation.
  - `multi_otsu()`: Multi-threshold Otsu segmentation.
  - `check_green()`: Determines if an object is green based on HSV filtering.
  - `lambsquarters()`: Specific segmentation method for a plant species.

---
### 4. `segment_utils.py`
Provides helper functions for segmentation tasks.

**Main Components:**
- `GenCutoutProps`: Computes various properties of segmented cutouts, including blur effects, color distribution, and descriptive statistics.
- Functions for calculating mean and standard deviation of RGB channels.
- Functions for connected component analysis.
- Color assignment functions for species labeling.

---
### 5. `utils.py`
General-purpose utility functions for JSON handling, image processing, and thresholding.

**Main Components:**
- `read_json()`, `flatten_json()`: Functions for handling and manipulating JSON files.
- `make_exg()`, `thresh_vi()`: Compute vegetation indices and apply thresholding.
- `reduce_holes()`: Morphological operations to clean segmentation masks.
- `match_season_to_date_ranges()`: Matches seasons to date ranges for dataset organization.
- `apply_mask()`: Applies a mask to an image for visualization or processing.
- `calculate_bbox_area_cm2()`: Computes the real-world area of bounding boxes.

---
## Usage
1. **Dataset Handling:** Use `datasets.py` to structure and manage image metadata and bounding boxes.
2. **Model Training and Inference:** Utilize `model.py` for deep learning model training and prediction.
3. **Segmentation:** Use `segment_species.py` and `segment_utils.py` for species identification and segmentation.
4. **General Utilities:** Leverage `utils.py` for various helper functions like JSON handling, image processing, and vegetation index calculations.

---
## Dependencies
This codebase requires the following libraries:
- `numpy`
- `pandas`
- `torch`
- `pytorch_lightning`
- `cv2`
- `skimage`
- `shapely`
- `segmentation_models_pytorch`

Install them using:
```sh
pip install numpy pandas torch pytorch-lightning opencv-python scikit-image shapely segmentation-models-pytorch
```

---
## Notes
- Ensure that images and metadata files are stored correctly before processing.
- The segmentation methods assume well-formatted input data.
- Logging is integrated into the scripts for debugging and performance tracking.
