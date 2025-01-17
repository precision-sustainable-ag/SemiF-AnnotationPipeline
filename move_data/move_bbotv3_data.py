import os  
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def copy_jpg_file(src, dest):
    """Copy a single ARW file from src to dest."""
    shutil.copy2(src, dest)

def copy_from_lockers_in_parallel(src_dir, dest_dir, max_workers=8):
    """Copy all ARW files in parallel from NFS to local storage."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # Collect all ARW file paths
    jpg_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(src_dir)
        for file in files if file.lower().endswith('.jpg')
    ]

    # Use ThreadPoolExecutor for parallel copying
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_jpg_file, jpg, os.path.join(dest_dir, os.path.basename(jpg))) for jpg in jpg_files]
        for future in as_completed(futures):
            future.result()  # Capture any exceptions


def main(cfg: DictConfig) -> None:
    batch = cfg.general.batch_id
    src = Path(cfg.data.longterm_storage2, "semifield-developed-images", batch, "images")
    dest = Path(cfg.data.batchdir, "images")
    dest.mkdir(parents=True, exist_ok=True)

    assert Path(src).exists(), f"Source directory {src} does not exist. Check the batch name."

    imgs = [img for img in src.glob("*.jpg")]
    log.info(f"Found {len(imgs)} ARW files in {src}")
    log.info(f"Copying from {src} to {dest}")

    copy_from_lockers_in_parallel(src, dest)






    