# Update Image EXIF Metadata

## Overview
This repository contains scripts for extracting, updating, and managing EXIF metadata in image files. The primary script, `update_image_exif.py`, modifies EXIF metadata for images by updating focal length information and image dimensions. The `utils.py` script provides utility functions for handling datasets, filtering dates, managing batch logs, and processing cutouts.

## Tree Structure
```
.
├─update_image_exif.py
└─utils.py
```

## Features
- Extract EXIF metadata from images.
- Update focal length and image dimensions in EXIF metadata.
- Estimate 35mm equivalent focal length based on sensor dimensions.
- Process multiple images concurrently for efficiency.
- Utility functions for batch processing, dataset management, and cutout processing.

---

## Scripts

### `update_image_exif.py`
This script is responsible for extracting and modifying EXIF metadata in image files.

#### Functions:
- `extract_exif(image_path)`: Extracts and displays EXIF metadata from an image.
- `estimate_focal_length_35mm(focal_length, sensor_width, sensor_height)`: Computes the equivalent focal length in 35mm format.
- `update_exif(image_path, focal_length, focal_length_35mm, width, height)`: Updates EXIF metadata with new focal length and image dimensions.
- `process_images(image_paths, focal_length, focal_length_35mm, width_pix, height_pix)`: Processes multiple images using concurrent execution.

#### Example Usage:
```python
image_dir = Path("data/semifield-developed-images/NC_2025-02-03/images")
images = sorted(image_dir.glob("*.jpg"))
exif_data = extract_exif(images[0])
print(exif_data)
```

---

### `utils.py`
A collection of utility functions for handling batch operations, metadata extraction, and data processing.

#### Functions:
- **Metadata Handling:**
  - `read_keys(keypath)`: Reads pipeline keys from a YAML configuration file.
  - `read_yaml(keypath)`: Reads configuration settings from a YAML file.
- **Batch Processing:**
  - `remove_batch(cfg, batch)`: Removes a batch entry from an unprocessed batch log.
  - `write_batch(cfg, batch)`: Writes a batch entry to a processed batch log.
- **Dataset Processing:**
  - `filter_and_select_dates(dates_list, start_date, end_date, state, num_dates)`: Filters and selects specific batch dates based on state and time range.
  - `cutout_csvs2df(cutout_dir)`: Loads and concatenates multiple cutout CSV files into a single DataFrame.
- **Image Processing:**
  - `trans_cutout(img)`: Converts an image with a black background to one with transparency.
  - `convert_to_dict(row)`: Converts bounding box information into a dictionary.
  - `chunk_list(lst, n)`: Splits a list into smaller chunks for parallel processing.

---

## License
This repository is licensed under the MIT License.

