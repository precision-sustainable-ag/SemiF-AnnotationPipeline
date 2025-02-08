"""
This script is intended to be run after manual validation of the images. It will:
1. update the metadata files to set the 'validated' key to True,
2. copy the full-sized and cutout data to the long-term storage (LTS) directory, and
3. optionally remove the local directories after successful transfer.

The script will prompt the user for confirmation twice, before proceeding with the updates and before removing local directories.
"""

import json
import logging
from pathlib import Path
import shutil
import re
import getpass
from omegaconf import DictConfig

USER_NAME = getpass.getuser()
log = logging.getLogger(__name__)
log.info(f"Script executed by user: {USER_NAME}")


class MetadataManager:
    """
    Handles operations on metadata JSON files.
    """

    @staticmethod
    def load_json(file_path: Path):
        """
        Loads JSON data from a file.

        Args:
            file_path (Path): Path to the JSON file.
        """
        with open(file_path, "r") as infile:
            return json.load(infile)

    @staticmethod
    def get_metadata_files(metadata_dir: Path):
        """
        Returns a sorted list of JSON metadata files from the given directory.

        Args:
            metadata_dir (Path): Directory containing metadata files.
        """
        return sorted(metadata_dir.glob("*.json"))

    @classmethod
    def update_validated_key(cls, metadata_file: Path):
        """
        Updates the 'validated' key in the metadata file to True.

        Args:
            metadata_file (Path): Path to the metadata file.
        """
        metadata = cls.load_json(metadata_file)
        metadata["validated"] = True
        return metadata

    @staticmethod
    def save_metadata(metadata: dict, metadata_file: Path):
        """
        Saves a metadata dictionary back to a JSON file.

        Args:
            metadata (dict): Metadata dictionary.
            metadata_file (Path): File path to save the metadata.
        """
        with open(metadata_file, "w") as outfile:
            json.dump(metadata, outfile, indent=4)

    @classmethod
    def update_metadata(cls, metadata_dir: Path):
        """
        Updates the 'validated' key for all JSON metadata files in the directory.

        Args:
            metadata_dir (Path): Directory containing metadata files.
        """
        metadata_files = cls.get_metadata_files(metadata_dir)
        for metadata_file in metadata_files:
            updated_metadata = cls.update_validated_key(metadata_file)
            cls.save_metadata(updated_metadata, metadata_file)
        log.info(f"Updated 'validated' key in {len(metadata_files)} metadata files.")
        return metadata_files


class DataMover:
    """
    Handles file and directory copying operations.
    """

    @staticmethod
    def is_valid_cutout_folder(folder_name: str):
        """
        Checks if the folder name matches the expected pattern AA_YYYY-MM-DD.

        Args:
            folder_name (str): Folder name to validate.
        """
        pattern = r"^[A-Z]{2}_\d{4}-\d{2}-\d{2}$"
        return bool(re.match(pattern, folder_name))

    @staticmethod
    def copy_dir(src: Path, dest: Path, check_existing: bool = False, file_extension: str = None, is_cutout: bool = False):
        """
        Copies an entire directory from src to dest.

        If `check_existing` is True, compares files (by extension) before copying.
        If `is_cutout` is True, validates the destination folder name.

        Args:
            src (Path): Source directory.
            dest (Path): Destination directory.
            check_existing (bool): Whether to compare files and avoid redundant transfers.
            file_extension (str): File extension pattern (e.g., "*.jpg", "*.json").
            is_cutout (bool): Whether this directory is a cutout (folder name pattern check).
        """
        if not src.exists():
            log.warning(f"Source directory does not exist: {src}. Skipping copy.")
            return

        # Validate folder name if this is a cutout directory.
        if is_cutout and not DataMover.is_valid_cutout_folder(dest.name):
            log.error(f"Destination directory '{dest}' does not match the expected cutout pattern (AA_YYYY-MM-DD). Skipping copy.")
            return

        # If checking for existing files, only copy missing files.
        if check_existing and file_extension:
            if not dest.exists():
                dest.mkdir(parents=True, exist_ok=True)

            existing_files = {f.name for f in dest.glob(file_extension)}
            local_files = {f.name for f in src.glob(file_extension)}
            if existing_files == local_files:
                log.info(f"Files already exist in LTS directory: {dest}. Skipping copy.")
                return
            else:
                log.info(f"Updating missing files in {dest}...")
                for file in src.glob(file_extension):
                    if file.name not in existing_files:
                        shutil.copy2(file, dest / file.name)
                return

        # Delete the destination if it exists (with a basic safeguard against deleting high-level directories).
        if dest.exists():
            if dest.is_dir() and len(dest.parts) > 6:
                if not all(keyword in dest.parts for keyword in ["screberg", "semifield", "longterm", "GROW", "research", "raatwell"]):
                    log.info(f"Removing existing destination directory: {dest}")
                    shutil.rmtree(dest)
            else:
                log.warning(f"Skipping deletion: {dest} seems too general.")

        log.info(f"Copying {src} to {dest}")
        shutil.copytree(src, dest)

    @staticmethod
    def verify_copy_success(src: Path, dest: Path, file_extension: str):
        """
        Verifies that the number of files copied from src to dest matches.

        Args:
            src (Path): Source directory.
            dest (Path): Destination directory.
            file_extension (str): File extension pattern to check.
        """
        if not src.exists():
            log.error(f"Source directory does not exist: {src}")
            return False

        if not dest.exists():
            log.error(f"Destination directory does not exist: {dest}")
            return False

        src_files = list(src.glob(file_extension))
        dest_files = list(dest.glob(file_extension))
        if len(src_files) == len(dest_files):
            log.info(f"Verification successful: {len(src_files)} files copied from {src} to {dest}.")
            return True
        else:
            log.warning(f"Verification failed: {len(src_files)} files in {src}, but {len(dest_files)} in {dest}.")
            return False

    def copy_fullsized_data(self, lts_dir: Path, batch_id: str, images: Path, metadata_dir: Path,
                            plant_dects_dir: Path, reference_dir: Path, semantic_mask_dir: Path):
        """
        Copies full-sized batch data to the LTS directory, performing file checks as needed.

        Args:
            lts_dir (Path): Root LTS directory.
            batch_id (str): Batch identifier.
            images (Path): Directory of images.
            metadata_dir (Path): Directory of metadata files.
            plant_dects_dir (Path): Directory of plant detection files.
            reference_dir (Path): Directory of reference images.
            semantic_mask_dir (Path): Directory of semantic mask files.
        """
        lts_developed_batch_dir = lts_dir / "semifield-developed-images" / batch_id
        lts_developed_batch_dir.mkdir(parents=True, exist_ok=True)
        copied = []

        # Copy images with file-checking to avoid unnecessary transfers.
        images_dest = lts_developed_batch_dir / "images"
        self.copy_dir(images, images_dest, check_existing=True, file_extension="*.jpg")
        copied.append(self.verify_copy_success(images, images_dest, "*.jpg"))

        # Copy metadata.
        metadata_dest = lts_developed_batch_dir / "metadata"
        self.copy_dir(metadata_dir, metadata_dest)
        copied.append(self.verify_copy_success(metadata_dir, metadata_dest, "*.json"))

        # Copy plant detection files.
        plant_dects_dest = lts_developed_batch_dir / "plant_detections"
        self.copy_dir(plant_dects_dir, plant_dects_dest)
        copied.append(self.verify_copy_success(plant_dects_dir, plant_dects_dest, "*.json"))

        # Copy reference images.
        reference_dest = lts_developed_batch_dir / "reference"
        reference_dest.mkdir(parents=True, exist_ok=True)
        self.copy_dir(reference_dir, reference_dest)
        copied.append(self.verify_copy_success(reference_dir, reference_dest, "*.csv"))

        # Copy semantic masks.
        semantic_dest = lts_developed_batch_dir / "meta_masks" / "semantic_masks"
        self.copy_dir(semantic_mask_dir, semantic_dest)
        copied.append(self.verify_copy_success(semantic_mask_dir, semantic_dest, "*.png"))

        if all(copied):
            log.info(f"Full-sized batch data successfully copied to LTS directory: {lts_developed_batch_dir}")
            return True
        else:
            log.warning("Full-sized batch data copy failed. Check logs for details.")
            return False

    def copy_cutout_data(self, lts_dir: Path, batch_id: str, cutout_dir: Path):
        """
        Copies cutout batch data to the LTS directory, ensuring the folder name is valid.

        Args:
            lts_dir (Path): Root LTS directory.
            batch_id (str): Batch identifier.
            cutout_dir (Path): Directory containing cutout data.
        """
        lts_cutout_batch_dir = lts_dir / "semifield-cutouts" / batch_id
        lts_cutout_batch_dir.mkdir(parents=True, exist_ok=True)
        # Validate cutout directory pattern before copying.
        self.copy_dir(cutout_dir, lts_cutout_batch_dir, is_cutout=True, check_existing=True, file_extension="*")
        return self.verify_copy_success(cutout_dir, lts_cutout_batch_dir, "*")


class BatchDataProcessor:
    """
    Orchestrates the batch data processing:
      - Updates metadata files
      - Copies full-sized and cutout data to long-term storage (LTS)
      - Optionally removes local directories after confirmation
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the processor with the configuration.

        Args:
            cfg (DictConfig): Configuration object.
        """
        self.cfg: DictConfig = cfg
        self.lts_dir: Path = Path(cfg.data.longterm_storage2)
        self.batch_id: str = cfg.general.batch_id

        # Full-sized data directories.
        self.images: Path = Path(cfg.batchdata.images)
        self.metadata_dir: Path = Path(cfg.batchdata.metadata)
        self.plant_dects_dir: Path = Path(cfg.batchdata.plant_dects)
        self.reference_dir: Path = Path(cfg.batchdata.autosfm) / "reference"
        self.semantic_mask_dir: Path = Path(cfg.batchdata.meta_masks) / "semantic_masks"

        # Cutout data directory.
        self.cutout_dir: Path = Path(cfg.batchdata.cutouts)

        # Local directories for removal confirmation.
        self.developed_src: str = cfg.data.batchdir
        self.cutout_src: str = cfg.batchdata.cutouts

        # Instantiate managers.
        self.metadata_manager = MetadataManager()
        self.data_mover = DataMover()
    
    def check_file_consistency(self) -> bool:
        """
        Checks if the number of metadata (.json), semantic mask (.png), and image (.jpg) files are equal.
        If not, prompts the user to decide if processing should proceed anyway.

        Returns:
            bool: True if counts are consistent or if user decides to proceed; False otherwise.
        """
        meta_count = len(list(self.metadata_dir.glob("*.json")))
        mask_count = len(list(self.semantic_mask_dir.glob("*.png")))
        image_count = len(list(self.images.glob("*.jpg")))
        log.info(f"File counts -- Metadata: {meta_count}, Masks: {mask_count}, Images: {image_count}")

        if meta_count != mask_count or meta_count != image_count:
            log.warning(f"File counts are not equal: images: {image_count}, metadata: {meta_count}, masks: {mask_count}")
            while True:
                user_input = input("Do you want to proceed anyway? (yes/no): ").strip().lower()
                if user_input in ["yes", "y"]:
                    log.info("User chose to proceed despite inconsistent file counts.")
                    return True
                elif user_input in ["no", "n"]:
                    log.warning("User aborted processing due to inconsistent file counts.")
                    return False
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
        return True

    @staticmethod
    def get_user_confirmation(developed_src: str = None, cutout_src: str = None, confirm_local_removal: bool = False):
        """
        Prompts the user for confirmation.

        Args:
            developed_src (str): Path to the developed batch directory (for removal confirmation).
            cutout_src (str): Path to the cutout directory (for removal confirmation).
            confirm_local_removal (bool): If True, confirms removal of local directories.
        """
        if confirm_local_removal:
            prompt = (f"\nAre you sure you want to remove these directories?\n"
                      f"1. developed - {developed_src}\n2. cutouts - {cutout_src}\n(yes/no): ")
            action = "local batch removal"
        else:
            prompt = "\nHave you manually inspected and validated all images? (yes/no): "
            action = "updates"

        while True:
            user_input = input(prompt).strip().lower()
            if user_input in ["yes", "y"]:
                log.info(f"User ({USER_NAME}) confirmed. Proceeding with {action}...")
                return True
            elif user_input in ["no", "n"]:
                log.warning(f"User ({USER_NAME}) canceled {action}. Exiting...")
                return False
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    def process(self):
        """
        Executes the overall batch processing pipeline.
          1. Prompts the user for confirmation.
          2. Updates metadata for both full-sized and cutout data.
          3. Copies full-sized and cutout data to the LTS directory.
          4. Optionally removes local directories after successful transfer.
        """
        # Check that the file counts in metadata, masks, and images are consistent.
        if not self.check_file_consistency():
            log.info("Aborting processing due to inconsistent file counts.")
            return
        
        # Confirm that the user has manually inspected and validated the images.
        if not self.get_user_confirmation():
            log.info(f"Pipeline halted by user ({USER_NAME}).")
            return

        # Update metadata in both directories.
        MetadataManager.update_metadata(self.metadata_dir)
        MetadataManager.update_metadata(self.cutout_dir)

        # Transfer full-sized data.
        fullsized_copied = self.data_mover.copy_fullsized_data(
            self.lts_dir,
            self.batch_id,
            self.images,
            self.metadata_dir,
            self.plant_dects_dir,
            self.reference_dir,
            self.semantic_mask_dir
        )

        # Transfer cutout data.
        cutout_copied = self.data_mover.copy_cutout_data(self.lts_dir, self.batch_id, self.cutout_dir)

        if fullsized_copied and cutout_copied:
            log.info("Batch data successfully transferred to LTS directory. Ready for local removal.")
            # Confirm removal of local directories.
            if not self.get_user_confirmation(self.developed_src, self.cutout_src, confirm_local_removal=True):
                log.info(f"Pipeline halted by user ({USER_NAME}). Exiting...")
                return

            log.info(f"Removing developed batch directory: {self.developed_src}")
            # Uncomment the next line to enable removal:
            shutil.rmtree(self.developed_src)

            log.info(f"Removing cutout batch directory: {self.cutout_src}")
            # Uncomment the next line to enable removal:
            shutil.rmtree(self.cutout_src)
        else:
            log.error("Data transfer to LTS directory failed. Local directories will not be removed.")


def main(cfg: DictConfig):
    """
    Main function to execute the batch data processing pipeline.

    Args:
        cfg (DictConfig): Configuration object.
    """
    processor = BatchDataProcessor(cfg)
    processor.process()


if __name__ == "__main__":
    main()
