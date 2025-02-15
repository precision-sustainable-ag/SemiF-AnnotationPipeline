# Database Management for Image Metadata

## Overview
This script, `database.py`, is responsible for managing a SQLite database that stores metadata for developed images and cutout images. It allows for table creation, bulk data insertion, and efficient querying of metadata related to image processing workflows. The script supports multiprocessing for handling large datasets efficiently.

## Features
- **Creates tables for storing developed images and cutouts metadata**
- **Bulk inserts metadata from JSON files into the database**
- **Parallel processing for improved performance**
- **Skips processing of predefined batches as per configuration**
- **Handles structured data storage for metadata attributes such as exif metadata, camera info, cutout properties, and annotations**
- **Optimized database operations to minimize redundant processing**

## Setup & Requirements
### Dependencies
Ensure you have the following installed:
- Python 3.8+
- Required Python libraries:
  ```bash
  pip install sqlite3 pandas tqdm hydra-core omegaconf
  ```

### Configuration
The script requires a configuration YAML file that defines database paths, batch settings, and JSON metadata locations.

## Usage
### Running the script
Execute the script with Hydra configuration:
```bash
python database.py
```
By default, it processes all batches unless configured otherwise.

### Configuration Parameters
The script utilizes a Hydra-configured YAML file with parameters such as:
- `db_path`: Path to the SQLite database.
- `skip_batches`: List of batch IDs to exclude from processing.
- `batch_size`: Number of records processed per batch.
- `developed_images`: Configuration for developed images table.
- `cutouts`: Configuration for cutouts table.

## Database Schema
### Developed Images Table
Stores metadata for processed full-size images.
```sql
CREATE TABLE developed_images (
    season TEXT,
    datetime TEXT,
    bbot_version TEXT,
    batch_id TEXT,
    image_id TEXT PRIMARY KEY,
    validated BOOLEAN,
    exif_meta TEXT,
    camera_info TEXT,
    annotations TEXT,
    categories TEXT,
    version TEXT
);
```

### Cutouts Table
Stores metadata for segmented plant cutouts.
```sql
CREATE TABLE cutouts (
    season TEXT,
    datetime TEXT,
    bbot_version TEXT,
    batch_id TEXT,
    image_id TEXT,
    cutout_id TEXT PRIMARY KEY,
    cutout_num INTEGER,
    cutout_height INTEGER,
    cutout_width INTEGER,
    lens_model TEXT,
    validated BOOLEAN,
    cutout_props TEXT,
    category TEXT,
    cutout_version TEXT
);
```

## Bulk Insertion
For efficient processing, the script reads JSON metadata files in chunks and inserts them into the database using multiprocessing.

To enable bulk insertion, modify the configuration file:
```yaml
bulk_insert: True
```

## Logs & Debugging
Logging is enabled to track table creation, data insertion, and errors.
```bash
tail -f database.log
```

## Notes
- Ensure correct paths to metadata JSON files in the configuration.
- Large datasets may require optimized batch sizes for faster processing.
- SQLite database is vacuumed after each session to optimize storage.

## Author
Developed as part of an image processing pipeline for structured metadata storage and retrieval.

---
This repository is part of a larger dataset processing workflow for agricultural imaging projects.

