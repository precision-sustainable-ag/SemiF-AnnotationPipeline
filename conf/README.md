# Configuration Overview

This document provides an overview of the configuration files used in the project. Each configuration file defines specific parameters necessary for processing, data management, and model execution.

## Directory Structure
```
.
├─asfm
│ └─asfm.yaml
├─config.yaml
├─convert
│ └─default.yaml
├─correct
│ └─correct.yaml
├─cvat
│ └─cvat.yaml
├─database
│ └─database.yaml
├─hydra
│ └─job_logging
│   └─custom.yaml
├─inspect
│ └─inspect.yaml
├─movedata
│ └─movedata.yaml
├─segment
│ └─segment.yaml
└─validate
  └─validate.yaml
```

---
## Configuration Files

### 1. **asfm.yaml** (AutoSfM Configuration)
Defines settings for the AutoSfM pipeline, including GPU usage, directory structure, and processing steps.
- **Pipeline Settings:** Defines steps such as resizing, adding photos, detecting markers, aligning cameras, and generating orthomosaics.
- **Processing Parameters:** Includes options for downscaling, alignment, depth maps, dense clouds, and model exports.
- **Output Directories:** Specifies locations for storing processed data, references, and reports.

### 2. **config.yaml** (Global Configuration)
Main configuration file that integrates multiple modules and controls overall pipeline behavior.
- **Hydra Defaults:** Specifies default configurations for different modules.
- **General Settings:** Defines metadata such as season, batch ID, and working directory.
- **Pipeline Controls:** Enables/disables specific processing steps such as downloading data, running AutoSfM, and classifying weeds.
- **Data Directories:** Defines storage locations for datasets, logs, and reference files.
- **Date Ranges:** Specifies collection periods for different datasets across locations (MD, NC, TX).
- **Model Paths:** Provides paths to classification models used for weed identification.

### 3. **convert/default.yaml** (Conversion Configuration)
Defines versions and testing parameters for full-sized images and cutouts.
- **Versions:** Tracks versioning for full-sized images and cutouts.
- **Testing Options:** Allows enabling sample tests and setting batch IDs.

### 4. **correct/correct.yaml** (Correction Parameters)
Defines start and end epochs for corrections, useful for model re-training or fine-tuning.

### 5. **cvat/cvat.yaml** (CVAT Annotation Settings)
Configures parameters for working with CVAT annotation tool.
- **Re-labeling Options:** Defines common names that should be re-labeled.
- **Image Processing:** Sets image sizes, sample count, and size classes.

### 6. **database/database.yaml** (Database Configuration)
Defines settings for connecting and managing the dataset database.
- **Database Path:** Specifies the location of the SQLite database.
- **Bulk Insert Paths:** Lists directories containing JSON metadata for bulk insert operations.
- **JSON Keys:** Specifies required keys from metadata files.

### 7. **hydra/job_logging/custom.yaml** (Custom Logging Configuration)
Sets up logging format, output files, and logging levels.
- **Log Handlers:** Configures console and file-based logging.
- **Output Directory:** Logs processing details in the batch-specific directory.

### 8. **inspect/inspect.yaml** (Inspection Settings)
Defines settings for visual inspection of cutouts.
- **Output Directory:** Sets storage locations for season-wise data.
- **Statistics and Compilation:** Controls whether to calculate and save statistics.
- **Visual Inspection:** Enables sampling of cutouts for manual review.

### 9. **movedata/movedata.yaml** (Data Movement Settings)
Controls downloading, uploading, and moving batch data.
- **Error Handling:** Logs missing and unprocessed data.
- **Azure Keys:** Specifies locations of SAS authentication keys.
- **Processing Steps:** Configures data checks, organizing missing data, and uploading batch results.

### 10. **segment/segment.yaml** (Segmentation Configuration)
Defines settings for segmentation using the SAM model.
- **Multiprocessing Controls:** Adjusts CPU and GPU allocation.
- **SAM Model Parameters:** Specifies batch size, model type, and checkpoint path.
- **Cutout Processing:** Enables multiprocessing for faster segmentation.

### 11. **validate/validate.yaml** (Validation Settings)
Defines parameters for dataset validation and visualization.
- **Sample Size:** Specifies the number of images to validate.
- **Plot Settings:** Configures visualization of bounding boxes, masks, and cutouts.
- **Storage Locations:** Sets directories for storing validation results.
- **Cutout Properties:** Defines thresholds for various cutout characteristics.

---
## Summary
These configuration files collectively define the pipeline's behavior, controlling data processing, segmentation, validation, and database interactions. The modular structure ensures flexibility and maintainability in the workflow.

