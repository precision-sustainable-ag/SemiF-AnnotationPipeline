# Download and Move Data Scripts

## Overview
This directory contains scripts and utilities for downloading and moving batch data from Azure Blob Storage and local storage. The scripts handle missing data checks, downloads, and batch organization for agricultural image processing.

## Directory Structure
```
.
├─ download_data.py
├─ move_bbotv3_data.py
└─ utils
   ├─ download_utils.py
   └─ list_batches.py
```

### Scripts

#### `download_data.py`
- Manages the download of batch data from Azure Blob Storage.
- Checks for missing files before initiating the download.
- Moves any empty or mismatched images and masks to an error directory.
- Logs execution time and potential errors.

#### `move_bbotv3_data.py`
- Copies developed image batches from long-term storage (NFS) to local storage.
- Uses parallel processing to speed up file transfers.
- Ensures the target directory exists before copying.

### Utilities (`utils/`)

#### `download_utils.py`
- Handles batch downloads by interfacing with Azure storage.
- Checks for missing data in the cloud and logs it.
- Moves empty or incorrectly paired image/mask files.

#### `list_batches.py`
- Lists and processes batch data in Azure Blob Storage.
- Identifies missing or incomplete batches.
- Saves batch processing status reports to a CSV file.

## Dependencies
- `omegaconf` for configuration management
- `logging` for error handling and execution logs
- `shutil`, `os`, `pathlib` for file operations
- `concurrent.futures` for parallel processing

Ensure that your `config.yaml` is properly set up with the necessary parameters before running the scripts.

