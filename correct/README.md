# Remove Data Script

## Overview
The `remove_data.py` script is designed to remove image-related data based on a specified epoch timestamp range. This includes metadata files, masks, cutouts, and references in associated metadata files and CSV files.

## Features
- Identifies images within the specified epoch timestamp range.
- Deletes associated metadata (`.json` files), masks (`.png` files), and cutout images.
- Updates metadata files to remove references to deleted cutouts.
- Removes rows from the batch CSV file containing deleted image IDs.
- Provides an option to preview masks before confirming deletion.

## Usage

### Configuration
The script reads parameters from a Hydra configuration file. Key parameters include:
- `start_epoch`: Start of the timestamp range.
- `end_epoch`: End of the timestamp range.

These parameters must be set before running the script.

### Confirmation Prompt
Before deletion, the script provides a list of masks that would be removed. You will be prompted to confirm deletion:
```bash
Do you want to proceed with deletion? (y/n):
```
If `y` or `yes` is entered, the deletion process will begin. Otherwise, the script exits without making changes.

## Important Notes
- Ensure that the `start_epoch` and `end_epoch` values are correctly set in the configuration.
- The deletion process is irreversible. Double-check the images and metadata before proceeding.
- The script logs all actions, and errors will be printed if files cannot be deleted.

## Dependencies
- Python 3.x
- `pandas`
- `hydra`
- `omegaconf`
- `pathlib`
- `json`
