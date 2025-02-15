# CVAT Data Packaging Script

## Overview
The `package_data.py` script is designed to process and package image cutouts and corresponding segmentation masks for CVAT annotation. The script applies image pre-processing steps, resizes images and masks, and prepares them in the required format for CVAT annotation import.

## Features
- Reads image cutouts and segmentation masks from a dataset.
- Filters images based on species and bounding box area.
- Resizes and tiles large images while padding smaller images to a fixed size (512x512 pixels).
- Stores processed images and masks in structured directories.
- Generates necessary annotation metadata, including label colors.
- Creates a compressed ZIP archive of the packaged dataset for easy upload to CVAT.

## Dependencies
- Python 3.7+
- OpenCV (`cv2`)
- Pandas
- NumPy
- Hydra (`omegaconf.DictConfig`)
- Logging
- Pathlib
- Shutil

## Configuration
The script uses a Hydra configuration file to define paths, batch settings, and other parameters. The relevant configurations are loaded dynamically and used throughout the processing pipeline.

## Usage
Run the script using Hydra with the appropriate configuration file:
```bash
python package_data.py
```

### Required Configurations:
Ensure the following configuration keys are properly set in `config.yaml`:
- `general.batch_id`: Specifies the batch ID being processed.
- `batchdata.images`: Path to the full-sized images directory.
- `batchdata.metadata`: Path to metadata files.
- `batchdata.meta_masks`: Directory containing semantic masks.
- `batchdata.cutouts`: Path to the cutout images directory.
- `cvat.cvat_stem`: Base directory name for CVAT output.
- `cvat.relabel_common_names`: List of species names to filter and relabel.
- `cvat.size_classes`: Bounding box size range filter.
- `cvat.sample_n`: Number of images to sample.
- `cvat.images_size`: Target image size for annotation.

## Output Structure
The processed dataset will be organized as follows:
```
./cvat/data/{batch_id}/
├── default/         # Processed images
├── defaultannot/    # Corresponding annotation masks
├── default.txt      # Image-mask pair list for CVAT
├── label_colors.txt # Label IDs and corresponding colors
└── {batch_id}_annotations_camvid.zip  # Packaged dataset archive
```

## Notes
- Ensure that all necessary input files exist before running the script.
- The script automatically removes excess samples if more than the required number are present.
- If an image or mask cannot be loaded, it will be skipped with a warning.
- The output dataset is structured to be directly compatible with CVAT's annotation format.

## License
This script is part of an internal dataset processing pipeline and follows the licensing terms of the project.

