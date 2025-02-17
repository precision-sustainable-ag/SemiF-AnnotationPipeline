import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import sys
import re
import json

log = logging.getLogger(__name__)

def remove_rows_from_csv(csv_path, image_ids):
    """Remove rows from a CSV file where the 'image_id' column matches any value in image_ids."""
    csv_path = Path(csv_path)

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)

            # Ensure 'image_id' column exists
            if "image_id" in df.columns:
                df_filtered = df[~df["image_id"].isin(image_ids)]
                df_filtered.to_csv(csv_path, index=False)
                print(f"Updated CSV: {csv_path} (removed {len(df) - len(df_filtered)} rows)")
            else:
                print(f"Warning: 'image_id' column not found in {csv_path}")

        except Exception as e:
            print(f"Error processing CSV {csv_path}: {e}")
    else:
        print(f"CSV file does not exist: {csv_path}")

def delete_file(file):
    """Delete a single file."""
    try:
        file.unlink()
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Error deleting {file}: {e}")

def delete_files(cutout_dir: Path, image_id: str):
    """Delete files matching a given pattern."""
    files = list(cutout_dir.glob(f"{image_id}_*.*"))
    print(f"Deleting {len(files)} files ")
    if not files:
        print(f"No files found: {cutout_dir}/{image_id}_*.*")
        return
    for file in files:
        try:
            file.unlink()
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def get_image_ids_from_directory(images_dir, start_epoch, end_epoch):
    """Get image IDs from the 'images' directory that fall within the specified epoch range."""
    image_ids = []
    images_dir = Path(images_dir)

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return []

    for image_file in images_dir.glob("*.jpg"):
        match = re.match(r"([A-Z]+)_(\d+)", image_file.stem)  # Extract state abbreviation & epoch timestamp
        if match:
            state, epoch = match.groups()
            epoch = int(epoch)
            if start_epoch <= epoch <= end_epoch:
                image_ids.append(f"{state}_{epoch}")

    return sorted(image_ids)

def explore_masks(cfg, start_epoch, end_epoch):
    """Explore the mask images for image IDs within the given epoch timestamp range."""
    images_dir = Path(cfg.batchdata.images)
    mask_dir = Path(cfg.batchdata.meta_masks, "semantic_masks")

    image_ids = get_image_ids_from_directory(images_dir, start_epoch, end_epoch)

    if not image_ids:
        print(f"No images found in the range {start_epoch} - {end_epoch}.")
        return

    for image_id in image_ids:
        mask_file = mask_dir / f"{image_id}.png"
        if mask_file.exists():
            print(f"Will delete: {mask_file}")
        else:
            print(f"Mask not found: {mask_file}")
            

def update_metadata_files(metadata_dir, image_ids):
    """Update metadata files by removing references to deleted cutouts."""
    metadata_dir = Path(metadata_dir)

    if not metadata_dir.exists():
        print(f"Metadata directory not found: {metadata_dir}")
        return

    for meta_file in metadata_dir.glob("*.json"):
        try:
            with open(meta_file, "r") as f:
                data = json.load(f)
            modified = False
            # Remove cutout references from `annotations`
            if "annotations" in data:
                for annotation in data["annotations"]:
                    # Identify full cutout_id patterns to remove (image_id + _<number>)
                    cutout_patterns = [f"{image_id}_" for image_id in image_ids]

                    # Remove overlapping cutouts
                    if "overlapping_cutout_ids" in annotation:
                        filtered_cutouts = [
                            cutout for cutout in annotation["overlapping_cutout_ids"]
                            if not any(cutout.startswith(pattern) for pattern in cutout_patterns)
                        ]
                        if len(filtered_cutouts) != len(annotation["overlapping_cutout_ids"]):
                            
                            print(annotation["overlapping_cutout_ids"])
                            annotation["overlapping_cutout_ids"] = filtered_cutouts
                            
                            print(filtered_cutouts)
                            modified = True


            # Save only if changes were made
            # if modified:
            #     with open(meta_file, "w") as f:
            #         json.dump(data, f, indent=4)
            #     print(f"Updated metadata file: {meta_file}")


        except Exception as e:
            print(f"Error updating metadata file {meta_file}: {e}")


def remove_image_data(cfg, start_epoch, end_epoch):
    """Remove all image-related data for image IDs within the given epoch timestamp range."""
    # Define paths
    batch_id = cfg.general.batch_id
    images_dir = Path(cfg.batchdata.images)
    metadata_dir = Path(cfg.batchdata.metadata)
    mask_dir = Path(cfg.batchdata.meta_masks, "semantic_masks")
    cutout_dir = Path(cfg.batchdata.cutouts)
    csv_path = cutout_dir / f"{batch_id}.csv"

    image_ids = get_image_ids_from_directory(images_dir, start_epoch, end_epoch)
    
    if not image_ids:
        print(f"No images found in the range {start_epoch} - {end_epoch}.")
        return

    for image_id in image_ids:
        # Delete metadata, masks, and cutouts
        delete_file(metadata_dir / f"{image_id}.json")
        delete_file(mask_dir / f"{image_id}.png")
        
        delete_files(cutout_dir, image_id)
    
    # Remove references in other metadata files
    update_metadata_files(metadata_dir, image_ids)

    # Remove rows from CSV
    remove_rows_from_csv(csv_path, image_ids)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    start_epoch = cfg.correct.start_epoch #1667495929
    end_epoch = cfg.correct.end_epoch #1667496083
    
    if start_epoch is None or end_epoch is None:
        print("Please provide start and end epochs.")
        sys.exit(1)
    
    explore_masks(cfg, start_epoch, end_epoch)

    # Confirm deletion by asking for user input
    response = input("Do you want to proceed with deletion? (y/n): ")
    if response.lower() != "y" or response.lower() != "yes":
        print("Deletion cancelled.")
        sys.exit(0)

    # Remove all data for images in the given epoch range
    remove_image_data(cfg, start_epoch, end_epoch)

if __name__ == "__main__":
    main()
