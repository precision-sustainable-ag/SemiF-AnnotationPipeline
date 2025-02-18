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
import hydra
import subprocess

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
    def safe_to_remove_developed_data_dir(dir_path: Path) -> bool:
        """
        Checks that the directory's final name matches the expected batch_id pattern and does not
        contain any forbidden keywords.

        Args:
            dir_path (Path): The directory to check.

        Returns:
            bool: True if safe to remove, False otherwise.
        """
        if len(dir_path.parts) < 6:
            log.error(f"Directory path '{dir_path}' is too short.")
            return False
        
        # List of strings that the dir name must contain one of
        required = ["images", "metadata", "plant-detections", "reference", "semantic_masks", "asfm"]
        
        for word in required:
            if word.lower() == dir_path.name:
                log.error(f"Directory name '{dir_path.name}' contains forbidden keyword '{word}'.")
                return True
        
        return False
    
    @staticmethod
    def safe_to_remove_batch(dir_path: Path) -> bool:
        """
        Checks that the directory's final name matches the expected batch_id pattern and does not
        contain any forbidden keywords.

        Args:
            dir_path (Path): The directory to check.

        Returns:
            bool: True if safe to remove, False otherwise.
        """
        # The expected pattern is e.g., "AA_YYYY-MM-DD"
        pattern = r"^[A-Z]{2}_\d{4}-\d{2}-\d{2}$"
        if not re.match(pattern, dir_path.name):
            log.error(f"Directory name '{dir_path.name}' does not match the expected batch_id pattern.")
            return False
        
        if len(dir_path.parts) < 6:
            log.error(f"Directory path '{dir_path}' is too short.")
            return False
        
        forbidden = ["image", "screberg", "longterm", "GROW_DATA", "developed", "semifield", "semi", "cutout"]
        
        for word in forbidden:
            if word.lower() in dir_path.name.lower():
                log.error(f"Directory name '{dir_path.name}' contains forbidden keyword '{word}'.")
                return False
        
        return True

    @staticmethod
    def change_permissions(directory: Path):
        """
        Changes permissions recursively on the given directory so that anyone can read (and traverse) it.
        """
        if not directory.exists():
            log.error(f"Error: Directory '{directory}' does not exist!", exc_info=True)
            return

        # This command gives read permission to all files and ensures directories are executable.
        cmd = ["chmod", "-R", "a+rX", str(directory)]
        try:
            subprocess.run(cmd, check=True)
            log.debug(f"Permissions updated for '{directory}'.")
        except subprocess.CalledProcessError as e:
            log.error(f"Error updating permissions: {e}", exc_info=True)
        return
    
    @staticmethod
    def copy_cutout_dir_rsync(src: Path, dest: Path):
        """
        Copies a cutout directory from src to dest using rsync.
        """
        if not src.exists():
            log.warning(f"Source directory does not exist: {src}. Skipping copy.")
            return

        # Validate folder name if this is a cutout directory.
        if not DataMover.is_valid_cutout_folder(dest.name):
            log.error(f"Destination directory '{dest}' does not match the expected cutout pattern (AA_YYYY-MM-DD). Skipping copy.")
            return

        log.info(f"Copying {src} to {dest} using rsync. This may take a while...")
        
        # Create the destination directory if it doesn't exist.
        dest.mkdir(parents=True, exist_ok=True)
        
        rsync_command = [
            "rsync",
            "-avhW",          # archive mode (recurses, preserves symlinks, times, perms, etc.), verbose, human-readable
            "--no-owner",    # do not preserve owner information
            "--no-group",    # do not preserve group information
            "--info=progress2",
            f"{src}/",       # trailing slash to copy contents of src
            str(dest)

        ]
        
        try:
            subprocess.run(rsync_command, check=True)
            log.info("rsync completed successfully.")
        except subprocess.CalledProcessError as e:
            log.error(f"rsync failed with error: {e}")

    @staticmethod
    def copy_dir(src: Path, dest: Path, images: bool = False, reference: bool = False):
        """
        Copies a directory from src to dest, optionally checking for existing files for 'images' folder.
        """
        if not src.exists():
            log.warning(f"Source directory does not exist: {src}. Skipping copy.")
            return

        # Avoid having to transfer large image files if they already exist in the LTS directory.
        if images:
            if dest.exists():
                existing_image_files = list(dest.glob("*.jpg"))
                local_image_files = list(src.glob("*.jpg"))
                existing_file_names = {f.name for f in existing_image_files}
                local_file_names = {f.name for f in local_image_files}
                
                if local_file_names == existing_file_names:
                    log.info(f"Files already exist in LTS directory: {dest}. Skipping copy.")
                    return
                else:
                    log.info(f"Updating missing files in {dest}...")
                    for file in local_image_files:
                        if file.name not in existing_file_names:
                            shutil.copy2(file, dest / file.name)
                    return
            else:
                shutil.copytree(src, dest)
                return
        else:
            # Delete the destination if it exists (with a basic safeguard against deleting high-level directories).
            if dest.exists():
                if DataMover.safe_to_remove_developed_data_dir(dest):
                    log.info(f"Removing existing destination directory: {dest}")
                    shutil.rmtree(dest)
                    if reference and (dest.parent / "autosfm").exists():
                        asfm = dest.parent / "autosfm"
                        
                        if DataMover.safe_to_remove_developed_data_dir(asfm):
                            log.info(f"Removing existing autosfm directory: {asfm}")
                            shutil.rmtree(asfm)
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
        self.copy_dir(images, images_dest, images=True)
        copied.append(self.verify_copy_success(images, images_dest, "*.jpg"))

        # Copy metadata.
        metadata_dest = lts_developed_batch_dir / "metadata"
        self.copy_dir(metadata_dir, metadata_dest)
        copied.append(self.verify_copy_success(metadata_dir, metadata_dest, "*.json"))

        # Copy plant detection files.
        plant_dects_dest = lts_developed_batch_dir / "plant-detections"
        self.copy_dir(plant_dects_dir, plant_dects_dest)
        copied.append(self.verify_copy_success(plant_dects_dir, plant_dects_dest, "*.json"))

        # Copy reference images.
        reference_dest = lts_developed_batch_dir / "reference"
        reference_dest.mkdir(parents=True, exist_ok=True)
        self.copy_dir(reference_dir, reference_dest, reference=True)
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
        # Validate cutout directory pattern before copying.
        self.copy_cutout_dir_rsync(cutout_dir, lts_cutout_batch_dir)
        self.change_permissions(lts_cutout_batch_dir)
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
        self.primary_lts_dir = Path(cfg.data.longterm_storage)
        self.secondary_lts_dir = Path(cfg.data.GROW_DATA)
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
            if self.get_user_confirmation(text="File counts are inconsistent. Do you want to proceed anyway?", action="processing"):
                return True
            else:
                return False

    @staticmethod
    def get_user_confirmation(text: str = None, action: str = None) -> bool:
        """
        Prompts the user for confirmation.

        Args:
            developed_src (str): Path to the developed batch directory (for removal confirmation).
            cutout_src (str): Path to the cutout directory (for removal confirmation).
            confirm_local_removal (bool): If True, confirms removal of local directories.
        """
        while True:
            user_input = input(f"{text} (yes/no): ").strip().lower()
            if user_input in ["yes", "y"]:
                log.info(f"Proceeding with {action}...")
                return True
            elif user_input in ["no", "n"]:
                log.warning(f"{action} canceled. Exiting...")
                return False
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    def transfer_data(self, description: str, transfer_func, *args, **kwargs) -> bool:
        if self.get_user_confirmation(text=f"Proceed with {description} data transfer?", action="data transfer"):
            result = transfer_func(*args, **kwargs)
            if not result:
                log.error(f"{description} data transfer failed.")
            return result
        else:
            log.info(f"Skipping {description} data transfer.")
            return False
        
    def find_lts_dir(self, cutouts: bool = False) -> Path:
        """
        Finds the correct LTS directory based on the batch ID and whether it has an "images" folder with images.
        """
        
        if cutouts:
            primary_images = self.primary_lts_dir / "semifield-cutouts" / self.batch_id
            secondary_images = self.secondary_lts_dir / "semifield-cutouts" / self.batch_id
            third_images = self.lts_dir / "semifield-cutouts" / self.batch_id
        else:
            primary_images = self.primary_lts_dir / "semifield-developed-images" / self.batch_id / "images"
            secondary_images = self.secondary_lts_dir / "semifield-developed-images" / self.batch_id / "images"
            third_images = self.lts_dir / "semifield-developed-images" / self.batch_id / "images"

        if primary_images.exists() and any(primary_images.iterdir()):
            log.info(f"Found images in primary LTS directory: {self.primary_lts_dir}")
            return self.primary_lts_dir
        elif secondary_images.exists() and any(secondary_images.iterdir()):
            log.info(f"Found images in secondary LTS directory: {self.secondary_lts_dir}")
            return self.secondary_lts_dir
        else:
            log.info(f"Using third LTS directory: {self.lts_dir}")
            return self.lts_dir

    def update_metadata(self):
        """
        Updates the 'validated' key in metadata files.
        """
        text = "Have you manually inspected and validated all images?"
        if not self.get_user_confirmation(text=text, action="metadata update" ):
            log.info(f"Pipeline halted by user ({USER_NAME}).")
            return
        # Update metadata for both full-sized and cutout data.
        for m_dir in [self.metadata_dir, self.cutout_dir]:
            MetadataManager.update_metadata(m_dir)
        
    def remove_cutouts_if_needed(self, cutout_lts_dir: Path) -> bool:
        """
        Checks for an existing LTS cutout directory and prompts for removal.
        """
        cutout_lts_dir = cutout_lts_dir / "semifield-cutouts" / self.batch_id
        cutout_removal_ok = True
        if cutout_lts_dir.exists() and any(cutout_lts_dir.iterdir()):
            log.info(f"Found existing LTS cutout directory: {cutout_lts_dir}")
            if self.data_mover.safe_to_remove_batch(cutout_lts_dir):
                if self.get_user_confirmation(text=f"LTS cutout directory '{cutout_lts_dir}' already exists. Remove it?", action="cutout removal"):
                    log.info(f"Removing existing LTS cutout directory: {cutout_lts_dir}")    
                    shutil.rmtree(cutout_lts_dir)
                    log.info(f"Removed existing LTS cutout directory: {cutout_lts_dir}")
            
            else:
                log.info(f"LTS cutout directory '{cutout_lts_dir}' did not pass checks for removal. Skipping.")
                cutout_removal_ok = False    
            
        else:
            if cutout_lts_dir.exists():
                log.info(f"The existing LTS cutout directory found at: {cutout_lts_dir} is empty. Skipping removal.")
            else:
                log.info(f"No existing LTS cutout directory found at: {cutout_lts_dir}.")
            cutout_removal_ok = True
        
        return cutout_removal_ok
    
    def confirm_and_remove_local_dirs(self) -> bool:
        """
        Prompts the user for confirmation and removes the local directories if confirmed.
        
        Returns:
            bool: True if directories were removed, False otherwise.
        """
        text = (f"\nAre you sure you want to remove these directories?\n"
                f"1. developed - {self.developed_src}\n"
                f"2. cutouts - {self.cutout_src}\n")
        
        if not self.get_user_confirmation(text=text, action="local directory removal"):
            log.info(f"Pipeline halted by user ({USER_NAME}). Exiting...")
            return False

        log.info(f"Removing developed batch directory: {self.developed_src}")
        shutil.rmtree(self.developed_src)
        
        log.info(f"Removing cutout batch directory: {self.cutout_src}")
        shutil.rmtree(self.cutout_src)
        
        return True
    
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
            return
        
        # Update metadata files.
        self.update_metadata()
        
        # Transfer full-sized data.
        # lts_dir = self.find_lts_dir()
        # fullsized_copied = self.transfer_data(
        #     "semifield-developed",
        #     self.data_mover.copy_fullsized_data,
        #     lts_dir,
        #     self.batch_id,
        #     self.images,
        #     self.metadata_dir,
        #     self.plant_dects_dir,
        #     self.reference_dir,
        #     self.semantic_mask_dir
        # )

        # Check for existing LTS cutout directory and prompt for removal
        cutout_lts_dir = self.find_lts_dir(cutouts=True)
        cutout_removal_ok = self.remove_cutouts_if_needed(cutout_lts_dir)
        
        # Transfer cutout data if allowed.
        if cutout_removal_ok:
            cutout_copied = self.transfer_data(
                "semifield-cutout",
                self.data_mover.copy_cutout_data,
                cutout_lts_dir,
                self.batch_id,
                self.cutout_dir
            )
        
        else:
            cutout_copied = False
        fullsized_copied = True
        
        if fullsized_copied and cutout_copied:
            self.confirm_and_remove_local_dirs()
        else:
            log.error("Data transfer to LTS directory failed. Local directories will not be removed.")

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
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
