import json
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from semif_utils.cutout_cleaner import CutoutMetadataCleaner
from semif_utils.fullsized_cleaner import FullsizedMetadataCleaner

from collections import defaultdict
import random
import pandas as pd
from pprint import pprint

log = logging.getLogger(__name__)

def flatten_json(nested_json, separator='_', prefix=''):
    """Recursively flatten a nested JSON."""
    flattened = {}
    for key, value in nested_json.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_json(value, separator, new_key))
        else:
            flattened[new_key] = value
    return flattened

def process_json_folder_to_csv(folder_path, output_csv_path):
    folder = Path(folder_path)
    all_data = []

    # Iterate over JSON files in the folder
    for json_file in folder.glob("*.json"):
        with open(json_file, 'r') as file:
            data = json.load(file)
            # Flatten nested JSON if necessary
            flattened_data = flatten_json(data)
            all_data.append(flattened_data)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save as CSV
    df.to_csv(output_csv_path, index=False)
    log.info(f"CSV saved at {output_csv_path}")

class BatchProcessor:
    def __init__(self, cfg: DictConfig, data_type ="semifield-developed-images") -> None:
        self.cfg = cfg
        self.batch_id = cfg.general.batch_id if not cfg.convert.test.enabled else cfg.convert.test.batch_id
        self.test_config = cfg.convert.test
        self.data_type = self._set_data_type(data_type)
        self.root_batch_dir = Path("data", self.data_type)
        self.sample_config = self.test_config.sample if self.test_config.enabled else None

    def _set_data_type(self, data_type: str) -> None:
        if self.test_config.enabled == True:
            self.data_type = "semifield-cutouts" if "cutout" in self.cfg.convert.test.data_type else "semifield-developed-images"
        else:
            self.data_type = data_type
        return self.data_type
        
    def _get_batch(self) -> list:
        # batches = sorted(self.root_batch_dir.glob("*"))
        batch = Path(self.root_batch_dir, self.batch_id)
        
        if self.data_type == "semifield-developed-images":
            # Filter out batches that do not have a metadata directory (but not if they're empty, that comes later)
            batch = Path(batch, "metadata")    
            
        return batch
    
    def _save_test_metadata(self, metadata_path: Path, cleaned_data: dict) -> None:
        test_dir = Path("data/test")
        if self.data_type == "semifield-cutouts":
            metadata_dir = test_dir / self.data_type / self.batch_id
        elif self.data_type == "semifield-developed-images":
            metadata_dir = test_dir / self.data_type / self.batch_id / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        output_path = metadata_dir / metadata_path.with_suffix(".json").name
        self._write_json(output_path, cleaned_data)
        return metadata_dir

    def _write_cleaned_metadata(self, metadata_path: Path, cleaned_data: dict) -> None:
        cleaned_dir = Path("data")
        if self.data_type == "semifield-cutouts":
            metadata_dir = cleaned_dir / self.data_type / self.batch_id
        elif self.data_type == "semifield-developed-images":
            metadata_dir = cleaned_dir / self.data_type / self.batch_id / "metadata"

        output_path = metadata_dir / metadata_path.name
        self._write_json(output_path, cleaned_data)
        return output_path.parent

    def _get_full_image_path(self, image_id: str, cutout_path: Path) -> Path:
        metadata_parent = str(cutout_path.parent).replace("semifield-cutouts", "semifield-developed-images")
        return Path(metadata_parent, "metadata", f"{image_id}.json")

    def _get_archive_full_image_paths(self, batch_path: Path, image_id: str) -> list:
        batch = batch_path.name
        batch_parent = str(batch_path.parent).replace("semifield-cutouts", "semifield-developed-images")
        archive_dirs = list(Path(batch_parent).glob("archive*"))
        return [Path(archive_dir, batch, "metadata", f"{image_id}.json") for archive_dir in archive_dirs if Path(archive_dir, batch, "metadata", f"{image_id}.json").exists()]

    def _get_archive_cutout_path(self, batch_path: Path, cutout_path: Path) -> list:
        batch = batch_path.name
        archive_dirs = list(Path(batch_path.parent).glob("archive*"))
        return [Path(archive_dir, batch, cutout_path.name) for archive_dir in archive_dirs]


    def _read_json(self, json_path: Path, suppress_error=False) -> dict:
        try:
            with open(json_path, "r") as json_file:
                return json.load(json_file)
        except FileNotFoundError as e:
            log.error(f"Could not read metadata file: {json_path} .")

        except json.JSONDecodeError as e:
            if not suppress_error:
                log.error(f"Could not decode JSON file: {json_path} .")
            return {}


    def _write_json(self, output_path: Path, data: dict) -> None:
        with open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def process_batches(self) -> None:
        batch = self._get_batch()
        
        log.info(f"Processing {batch.name} in {self.root_batch_dir}")
    
        if not batch.exists():
            log.error(f"Batch {batch} does not exist.")
            return
        
        if self.data_type == "semifield-cutouts":
            output_dir = self.process_cutout_batch(batch)
            if output_dir is not None:
                output_csv_path = output_dir / f"{batch.name}.csv"
                process_json_folder_to_csv(output_dir, output_csv_path)
        
        elif self.data_type == "semifield-developed-images":
            output_dir = self.process_fullsize_batch(batch)
        
        log.info(f"Finished processing batch: {batch.name} in {self.root_batch_dir}")
        log.info(f"Saved cleaned metadata to {output_dir}")
    
    def _get_unique_image_ids_from_cutouts(self, metadata_files: list) -> list:
        image_ids = []
        for metadata_path in tqdm(metadata_files, desc="Getting unique image IDs", leave=False):
            image_id = "_".join(metadata_path.stem.split("_")[:-1])
            if image_id not in image_ids:
                image_ids.append(image_id)
        return image_ids
    
    def _find_fullsized_metadata_for_cutout_cleaner(self, batch_path: Path, image_ids: list) -> dict:
        cached_data = {}
        # Create all the possible paths to find the fullsized metadata
        developed_root = str(self.root_batch_dir).replace("semifield-cutouts", "semifield-developed-images")
        
        
        # Same storage directory    
        full_meta_dir_same_storage = Path(developed_root, batch_path.name, "metadata")

        # Archive directories of the same storage directory and different storage directory
        reference_dir = self._get_reference_directory(batch_path)        
        dfov = self._load_fov_csv(reference_dir)

        # Iterate through unique image IDs
        for image_id in image_ids:
            # First check in the most obvious location
            if Path(full_meta_dir_same_storage, f"{image_id}.json").exists():
                full_metadata_path = full_meta_dir_same_storage / f"{image_id}.json"
                full_metadata = self._read_json(full_metadata_path, suppress_error=True)
                
                if full_metadata:
                    
                    exif_meta = full_metadata.get("exif_meta", {})
                    if "LensModel" in exif_meta and "ImageWidth" in exif_meta and "ImageLength" in exif_meta:
                        cached_data[image_id] = {
                            "exif_meta": {
                                "LensModel": exif_meta["LensModel"], 
                                "ImageWidth": exif_meta["ImageWidth"], 
                                "ImageLength": exif_meta["ImageLength"]
                                },
                                "full_metadata_path": full_metadata_path,
                                "fov": dfov.get(image_id, {})
                                }
                    else:
                        missing_fields = [field for field in ["LensModel", "ImageWidth", "ImageLength"] if field not in exif_meta]
                        log.error(f"Fullsized metadata for {image_id} in {full_metadata_path} is missing required fields ({missing_fields}) in same batch.")
            
                
        return cached_data
    
    def _get_reference_directory(self, batch_path: Path) -> Path:
        """Get the reference directory for a given batch path by checking multiple possible locations."""
        if self.data_type == "semifield-cutouts":
            possible_paths = [
                Path(str(batch_path).replace("semifield-cutouts", "semifield-developed-images"), "reference"),
                Path(str(batch_path).replace("semifield-cutouts", "semifield-developed-images"), "autosfm", "reference"),
            ]
            
            # Add fallback path without "autosfm" for all cases
            possible_paths.append(Path(str(possible_paths[1]).replace("autosfm/reference", "reference")))
        
        elif self.data_type == "semifield-developed-images":
            possible_paths = [
                Path(batch_path, "reference"),
                Path(batch_path, "autosfm", "reference"),
            ]
            
            # Add fallback path without "autosfm" for all cases
            possible_paths.append(Path(str(possible_paths[1]).replace("autosfm/reference", "reference")))
        

        for reference_dir in possible_paths:
            if reference_dir.exists():
                if Path(reference_dir, "fov.csv").exists():
                    return reference_dir
                
        log.error(f"Could not find reference directory for batch {batch_path}")
        return Path()  # Return an empty path if nothing is found
    
    def _load_fov_csv(self, reference_dir: Path) -> dict:
        if reference_dir.exists():
            return pd.read_csv(reference_dir / "fov.csv").set_index("label").to_dict(orient="index")
        return {}
    
    def _load_cam_reference_csv(self, reference_dir: Path) -> dict:
        if reference_dir.exists():
            df = pd.read_csv(reference_dir / "camera_reference.csv")
            df = df.drop_duplicates(subset="label", keep="first")
            return df.set_index("label").to_dict(orient="index")
        return {}

        
    def _cache_all_full_metadata_paths(self, metadata_files):
        """
        Caches metadata files from different storage locations and archives.

        Args:
            metadata_files (list of Path): List of paths to metadata files.
            batch_id (str): The batch ID to locate the relevant files.

        Returns:
            dict: A dictionary with image IDs as keys and a list of corresponding metadata paths as values.
        """
        
        # Initialize a defaultdict to automatically handle list appends
        cached_data = defaultdict(lambda: {"metadata_paths": []})
        
        # Cache metadata from the provided metadata files
        for metadata_path in metadata_files:
            image_id = metadata_path.stem
            cached_data[image_id]["metadata_paths"].append(metadata_path)
                
        # Convert defaultdict back to regular dictionary (optional, if you want a plain dict)
        return dict(cached_data)
    
    def process_fullsize_batch(self, batch_path: Path) -> Path:
        metadata_files = list(batch_path.glob("*.json"))

        if not metadata_files:
            log.warning(f"No metadata files found in {batch_path}")
            return None
        
        # Find all the metadata paths and put in a dictionary indexed by image ID
        reference_dir = self._get_reference_directory(batch_path.parent)
        dfov = self._load_fov_csv(reference_dir)   
        cam_reference = self._load_cam_reference_csv(reference_dir)     
        metadata_cache = self._cache_all_full_metadata_paths(metadata_files)
        cleaner: FullsizedMetadataCleaner = FullsizedMetadataCleaner(cfg=self.cfg, batch_path=batch_path)
        
        for metadata_path in tqdm(metadata_files, desc=f"Processing metadata for batch {batch_path.name}", leave=False):
            image_id = metadata_path.stem
            metadata_cache[image_id]["fov"] = dfov.get(image_id, {})
            metadata_cache[image_id]["camera_reference"] = cam_reference.get(image_id, {})
            image_data_cache = metadata_cache[image_id]
            
            # TODO: implement a way of using the multiple metadata_paths if they exists
            # Combine all the metadata you find and just pull from it what you need then organize it at the very end
            # Only disadvantage here is fear of overwriting data but that shouldn't be the case
            combined_metadata = {}
            for metadata_path_v_ in image_data_cache["metadata_paths"]:
                metadata = self._read_json(metadata_path_v_)
                combined_metadata.update(metadata)

            cleaned_metadata = cleaner.clean(metadata_path, combined_metadata, image_data_cache)
                
            
            if self.test_config.enabled:
                output_dir = self._save_test_metadata(metadata_path, cleaned_metadata)
            
            else:
                output_dir = self._write_cleaned_metadata(metadata_path, cleaned_metadata)

        return output_dir
    
    def process_cutout_batch(self, batch_path: Path) -> Path:
        metadata_files = list(batch_path.glob("*.json"))
        
        if not metadata_files:
            log.warning(f"No metadata files found in {batch_path}")
            return None
            
        image_ids = self._get_unique_image_ids_from_cutouts(metadata_files)
        cached_fullsized_data = self._find_fullsized_metadata_for_cutout_cleaner(batch_path, image_ids) # TODO: create a conditional to make sure the fullsized metadata exists, and if not, log the error and skip the processing or process without the needed information
        
        if not cached_fullsized_data:
            log.error(f"No fullsized metadata found for batch {batch_path}")
            return None
        # cached_data = self._cache_image_and_archived_cutout_paths(metadata_files, batch_path)
        cleaner = CutoutMetadataCleaner(cfg=self.cfg, batch_path=batch_path)
    
        for metadata_path in tqdm(metadata_files, desc=f"Processing cutouts for batch {batch_path.name}", leave=False):
            
            cutout_id = metadata_path.stem
            image_id = "_".join(cutout_id.split("_")[:-1])
    
            image_data = cached_fullsized_data[image_id]
            
            metadata = self._read_json(metadata_path)
            cleaned_metadata = cleaner.clean(
                metadata_path, metadata,
                image_data
            )
            
            if self.test_config.enabled:
                output_dir = self._save_test_metadata(metadata_path, cleaned_metadata)
            else:
                output_dir = self._write_cleaned_metadata(metadata_path, cleaned_metadata)

        return output_dir

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    batch_processor = BatchProcessor(cfg)
    batch_processor.process_batches()

if __name__ == "__main__":
    main()
