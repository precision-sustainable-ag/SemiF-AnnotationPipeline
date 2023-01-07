import logging
import time

from omegaconf import DictConfig

from move_data.utils.download_utils import DownloadData

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    start = time.time()
    batch = cfg.general.batch_id
    dd = DownloadData(cfg)

    try:
        # Makes sure azure source has all necessary data
        log.info("Finding missing azure data in blob container.")
        dd.find_missing_azure_data()
        if len(dd.miss_src) > 0:
            log.error(f"Azure data missing: {dd.miss_src}. Skipping batch.")
            exit(1)
    except Exception as e:
        log.exception(f"Failed to find missing Azure data for {batch}.\n{e}")
        exit(1)

    try:
        # Get the necessary data directories that are not present locally
        log.info(
            "Finding missing local data that needs to be downloaded from container."
        )
        dd.list_missing_local()
        log.info(f"Missing {dd.miss_local}.")
    except Exception as e:
        log.exception(
            f"Failed to find missing local data prior to download for {batch}.\n{e}"
        )
        exit(1)

    try:
        # Run download script
        log.info("Downloading data from azure blob container")
        # dd.download_azure_batch()
    except Exception as e:
        log.exception(f"Failed to download batch from blob container.\n{e}")
        exit(1)

    try:
        # Move files that have size 0
        log.info("Searching for empty images or masks.")
        dd.move_empty_data()
    except Exception as e:
        log.exception(f"Failed to move empty masks or images.\n{e}")
        exit(1)

    try:
        log.info("Searching for mismatched images and masks.")
        dd.move_missing_images()
    except Exception as e:
        log.exception(f"Moving mismatched images failed. Exiting. \n{e}")
        exit(1)

    end = time.time()
    log.info(f"Download completed in {(end - start)/60} minutes.")
