import logging
import os
import shutil
from pathlib import Path
from utils.utils import read_keys
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class DownloadData:

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.download_script = cfg.movedata.download_data.download_sh
        self.container_list = Path(cfg.movedata.find_missing.container_list)
        print(self.container_list)
        self.developed = cfg.data.developeddir
        self.batch = cfg.general.batch_id
        self.batchdir = cfg.data.batchdir
        self.error_data = cfg.movedata.error_data
        self.azure_items = cfg.movedata.azure_batch_contents
        self.list_azure_path = cfg.movedata.list_data
        self.miss_az_log = cfg.movedata.miss_azure_log
        self.necessary = cfg.movedata.download_data.input_data
        self.pkeys = read_keys(cfg.movedata.SAS_keys)

        self.miss_src = []
        self.miss_local = []

    def list_missing_local(self):
        """Checks for the existance of 2 necessary folders and 1 optional in the main local batch directory.
            1. images - (folder) preprocessed ("developed") images 
            2. masks - (folder) bench bot masks
            3. plant-detections (optional) - (folder) of plant-detection results (gives warning if not present)
        """

        # Check for necessary directories
        for nec in self.necessary:
            if not Path(self.batchdir, nec).exists():
                self.miss_local.append(nec)

    def log_missing_azure_data(self, missing_data):
        # Logs missing azure data to a log file in .batchlogs.
        if len(missing_data) > 0:
            log.error(f"Missing data in azure blob container found.")
            log.error(
                f"Loggging to {Path(self.miss_az_log).name} in .batchlogs")
            with open(self.miss_az_log, 'a+') as f:
                for dat in missing_data:
                    f.write(f"{self.batch}: {dat}\n")

    def list_azure_data(self):
        # Read local azure batch content file
        with open(self.container_list, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    def find_missing_azure_data(self):
        # Organizes and cleans the results of ListData.sh
        lines = self.list_azure_data()

        # Get lines only relevant to this batch
        batch_lines = [line for line in lines if self.batch in line]

        # Remove batch parent
        present_data = [
            data.replace(f"{self.batch}/", "") for data in batch_lines
        ]

        # Find difference between 2 sets
        for i in self.necessary:
            if i not in present_data:
                self.miss_src.append(i)
        # Ignore prediction_images because they aren't necessary for processing
        if "prediction_images" in self.miss_src:
            self.miss_src.remove("prediction_images")

        # Log if necessary
        self.log_missing_azure_data(self.miss_src)

    def download_azure_batch(self):
        # Run download script
        for i in self.miss_local:
            os.system(
                f'bash {self.download_script} {self.batch} {i} {self.developed} "{self.pkeys.down_dev}"'
            )

    def move_empty_data(self):
        # Move empty files to a seperate folder ("error_data")
        imgs = list(Path(self.batchdir, "images").glob("*.jpg"))
        masks = list(Path(self.batchdir, "masks").glob("*.png"))
        # Get file size for images and masks
        empty_imgs = [x for x in imgs if x.stat().st_size == 0]
        empty_masks = [x for x in masks if x.stat().st_size == 0]

        if len(empty_imgs) != 0:
            log.warning(
                f"Moving empty images: {len(empty_imgs)} to 'error_data'.")
            for src in empty_imgs:
                dst_dir = self.error_data
                dst = Path(dst_dir, "images", src.name)
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)

        if len(empty_masks) != 0:
            log.warning(
                f"Moving empty masks: {len(empty_masks)} to 'error_data'.")
            for src in empty_masks:
                dst_dir = self.error_data
                dst = Path(dst_dir, "masks", src.name)
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)

    def move_missing_images(self):
        """Check that the number of downloaded images and masks match.
        If they don't, move extra data to a "error_data" folder in the batch directory
        Assume either images or masks will have missing data, not both"""

        # TODO: account for if seperate data is missing in images and masks
        imgs = [
            x.stem for x in Path(self.batchdir, "images").glob("*.jpg")
            if ".pp3" not in x.name
        ]
        masks = [
            x.stem.replace("_mask", "")
            for x in Path(self.batchdir, "masks").glob("*.png")
        ]
        flag_suffix = ("masks", "_mask.png") if (len(imgs) < len(masks)) and (
            len(imgs) == len(masks)) else ("images", ".jpg")
        set1 = set(imgs)
        set2 = set(masks)

        missing = set(set1).symmetric_difference(set2)
        missing_name = [x + flag_suffix[1] for x in missing]
        src_dir = Path(self.batchdir, flag_suffix[0])

        miss_paths = [Path(src_dir, x) for x in missing_name]
        if len(miss_paths) > 10:
            log.error("More than 10 missing images were identified. Exiting.")
            exit(1)
        else:
            for src in miss_paths:
                log.warning(
                    f"Moving missing image: {len(miss_paths)} to 'error_data'."
                )
                dst_dir = Path(self.error_data)
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = Path(dst_dir, src.name)
                shutil.move(src, dst)
