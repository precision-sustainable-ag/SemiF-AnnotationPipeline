# Metadata Cleaning Scripts

## Overview
This repository contains Python scripts for cleaning and reformatting metadata files systematically. The scripts are designed to process metadata related to agricultural image datasets, ensuring consistency, completeness, and compliance with predefined schemas.

## File Structure
```
.
├── cutout_cleaner.py
└── fullsized_cleaner.py
```

## Dependencies
These scripts rely on the following Python libraries:
- `shutil`
- `logging`
- `pathlib`
- `pprint`
- `omegaconf`
- `json`
- `numpy`

Additionally, the scripts utilize helper functions from the `semif_utils.utils` module.

## Scripts Description

### 1. `cutout_cleaner.py`
This script defines the `CutoutMetadataCleaner` class, which is responsible for cleaning metadata related to cutout images. The cleaning process involves extracting key metadata fields, verifying their validity, and organizing them according to a predefined schema.

#### Key Features:
- Retrieves and validates root metadata properties (e.g., `season`, `datetime`, `batch_id`, `image_id`).
- Computes additional properties such as `bbox_area_cm2`.
- Organizes and structures metadata into categories and subcategories.
- Adds validation flags and version numbers.
- Handles missing metadata fields with appropriate logging.

#### Usage:
```python
from cutout_cleaner import CutoutMetadataCleaner
from pathlib import Path
from omegaconf import DictConfig

# Example configuration and batch path
cfg = DictConfig({"data": {"species": "path/to/species.json", "utilsdir": "path/to/utils"}, "date_ranges": {}})
batch_path = Path("/path/to/batch")

cleaner = CutoutMetadataCleaner(cfg, batch_path)
cleaned_metadata = cleaner.clean(metadata_path=Path("metadata.json"), metadata={}, image_data={})
```

### 2. `fullsized_cleaner.py`
This script defines the `FullsizedMetadataCleaner` and `AnnotationCleaner` classes, which clean metadata related to full-sized images and their annotations.

#### Key Features:
- Extracts and validates EXIF metadata (e.g., camera settings, exposure, lens information).
- Organizes annotation-related metadata, ensuring bounding boxes and class IDs are correct.
- Computes and formats field-of-view (FOV) data.
- Adds validation flags and organizes metadata keys based on a predefined schema.
- Replaces `NaN` values with `null` to maintain JSON integrity.

#### Usage:
```python
from fullsized_cleaner import FullsizedMetadataCleaner
from pathlib import Path
from omegaconf import DictConfig

# Example configuration and batch path
cfg = DictConfig({"data": {"species": "path/to/species.json", "utilsdir": "path/to/utils"}, "date_ranges": {}})
batch_path = Path("/path/to/batch")

cleaner = FullsizedMetadataCleaner(cfg, batch_path)
cleaned_metadata = cleaner.clean(metadata_path=Path("metadata.json"), metadata={}, image_data={})
```

## Logging
Both scripts utilize Python's `logging` module for error handling and debugging. Logs include warnings for missing or unexpected values and errors for critical missing fields.

## Customization
- The metadata schemas (`cutout_schema.json` and `fullsized_schema.json`) can be modified to update the required fields.
- Date range mappings and season identification logic can be adjusted in the configuration file.

## Future Improvements
- Implement parallel processing for batch metadata cleaning.
- Add support for additional metadata formats.
- Improve handling of edge cases in metadata validation.
