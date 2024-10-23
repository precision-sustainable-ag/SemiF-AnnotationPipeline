import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import logging
from omegaconf import DictConfig
from tqdm import tqdm
from semif_utils.metadata_schema import FULLSIZED_METADATA_SCHEMA
log = logging.getLogger(__name__)

class MetadataCleaner:
    """Class to handle the cleaning and reformatting of metadata."""

    @staticmethod
    def clean_metadata(data: dict) -> dict:
        """
        Clean and reformat the metadata dictionary by removing unnecessary keys 
        and renaming some keys.

        Args:
            data (dict): The original metadata dictionary.

        Returns:
            dict: The cleaned and reformatted metadata dictionary.
        """
        # Keys to remove from the metadata
        keys_to_remove = ["dap", "blob_home", "schema_version", "data_root", "image_path", "plant_date", "rel_path"]
        for key in keys_to_remove:
            data.pop(key, None)

        # Rename keys for consistency
        if "fullres_height" in data:
            data["height"] = data.pop("fullres_height")

        if "fullres_width" in data:
            data["width"] = data.pop("fullres_width")

        # Clean camera information
        camera_info = data.get("camera_info", {})
        camera_info_keys_to_remove = ["camera_location", "yaw", "pitch", "roll", "fov"]
        for key in camera_info_keys_to_remove:
            camera_info.pop(key, None)
        data["camera_info"] = camera_info

        # Clean EXIF metadata
        exif_meta = data.get("exif_meta", {})
        exif_keys_to_remove = [
            "PhotometricInterpretation", "SamplesPerPixel", "XResolution", "YResolution", "PlanarConfiguration",
            "ResolutionUnit", "Rating", "DateTimeOriginal", "DateTimeDigitized", "MeteringMode", "FileSource",
            "SceneType", "CustomRendered", "DigitalZoomRatio", "SceneCaptureType", "MakerNote", "ImageDescription",
            "UserComment", "ApplicationNotes", "Tag", "SubIFDs", "BitsPerSample", "Compression", "Orientation",
            "ExifOffset", "ExposureBiasValue"
        ]
        for key in exif_keys_to_remove:
            exif_meta.pop(key, None)
        data["exif_meta"] = exif_meta

        return data

class CoordinateConverter:
    """Class to handle the conversion of local coordinates to [x, y, width, height] format."""
    
    @staticmethod
    def calculate_bounding_box_area(global_coordinates):
        """
        This function takes a dictionary containing the global coordinates
        of a bounding box and returns its area in square meters.
        
        Args:
        global_coordinates (dict): A dictionary with the following keys:
            "top_left" (list): [x, y] coordinates of the top left corner.
            "top_right" (list): [x, y] coordinates of the top right corner.
            "bottom_left" (list): [x, y] coordinates of the bottom left corner.
            "bottom_right" (list): [x, y] coordinates of the bottom right corner.
            
        Returns:
        float: Area of the bounding box in square meters.
        """
        top_left = np.array(global_coordinates['top_left'])
        top_right = np.array(global_coordinates['top_right'])
        bottom_left = np.array(global_coordinates['bottom_left'])

        if top_left[0] == 0 and top_left[1] == 0 and top_right[0] == 0 and top_right[1] == 0 and bottom_left[0] == 0 and bottom_left[1] == 0:
            return None

        if len(top_left) != 2:
            top_left = top_left[:2]
        if len(top_right) != 2:
            top_right = top_right[:2]
        if len(bottom_left) != 2:
            bottom_left = bottom_left[:2]
            
        # Calculate the width (distance between top_left and top_right)
        width = np.linalg.norm(top_right - top_left)

        # Calculate the height (distance between top_left and bottom_left)
        height = np.linalg.norm(top_left - bottom_left)

        # Calculate the area of the bounding box
        area = width * height
        # Convert square meters to square centimeters
        area_cm2 = area * 10000
        return area_cm2
    
    @staticmethod
    def convert_to_xywh(local_coordinates: dict, image_width: int, image_height: int) -> List[int]:
        """
        Convert local coordinates to [x, y, width, height] format using the 
        dimensions of the image.

        Args:
            local_coordinates (dict): Local coordinates.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            List[int]: Converted coordinates in [x, y, width, height] format.
        """
        # Extract corner points
        top_left = local_coordinates["top_left"]
        top_right = local_coordinates["top_right"]
        bottom_left = local_coordinates["bottom_left"]

        # Calculate x, y, width, height
        x = int(top_left[0] * image_width)
        y = int(top_left[1] * image_height)
        width = int((top_right[0] - top_left[0]) * image_width)
        height = int((bottom_left[1] - top_left[1]) * image_height)
        return [x, y, width, height]

class CategoryManager:
    """Class to manage and reset categories in the metadata."""

    @staticmethod
    def load_species_info(path: str) -> dict:
        """
        Load species information from a JSON file located at the given path.

        Args:
            path (str): Path to the species JSON file.

        Returns:
            dict: Loaded species information.
        """
        log.debug(f"Loading species info from {path}.")
        with open(path, "r") as outfile:
            data = json.load(outfile)
        return data

    @staticmethod
    def reset_category(data: dict, path: str, batch_id: str, dt: str, date_ranges: dict) -> dict:
        """
        Reset category information in the metadata based on the species information 
        and date ranges.

        Args:
            data (dict): Metadata dictionary.
            path (str): Path to species JSON file.
            batch_id (str): Batch ID.
            dt (str): Datetime string.
            date_ranges (dict): Date ranges for crop types.

        Returns:
            dict: Updated metadata with reset category.
        """
        data["category"] = data.pop("cls", None)
        log.debug("Resetting category.")
        spec_info = CategoryManager.load_species_info(path)["species"]
        state_id = batch_id.split("_")[0]
        USDA_symbol = data["category"]["USDA_symbol"]
        class_id = data["category"]["class_id"]
        data["cutout_id"] = data.pop("bbox_id", None)
        cutout_id = data["cutout_id"]
        crop_year = DateRangeChecker.get_crop_type(state_id, dt, date_ranges).replace(" ", "_")
        

        if state_id == "MD" and crop_year == "weeds_2023" and USDA_symbol in ["ELIN3", "URPL2"]: # ELIN3 is Goosegrass and URPL2 is Broadleaf signalgrass
            log.warning(f'Changing {USDA_symbol} to unknown ("plant") for cutout: {batch_id}/{cutout_id}.json')
            log.critical(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['plant']['class_id']}")
            data["category"] = spec_info["plant"]
        
        # Adjust categories based on specific rules
        if state_id == "NC" and crop_year == "weeds_2022" and USDA_symbol == "URPL2":
            log.warning(f'Changing {USDA_symbol} to Texas Millet ("URTE2") for cutout: {batch_id}/{cutout_id}.json')
            log.critical(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['URTE2']['class_id']}")
            data["category"] = spec_info["URTE2"]

        # Adjust categories based on specific rules
        if state_id == "NC" and crop_year == "cover_crops_2023/2024" and USDA_symbol == "TRAE":
            log.warning(f'Changing {USDA_symbol} to unknown ("plant") for cutout: {batch_id}/{cutout_id}.json')
            log.critical(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['plant']['class_id']}")
            data["category"] = spec_info["plant"]

        if state_id == "TX" and crop_year == "weeds_2023" and USDA_symbol == "ECCO2":
            log.warning(f'Changing {USDA_symbol} to unknown ("plant") for cutout: {batch_id}/{cutout_id}.json')
            log.critical(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['plant']['class_id']}")
            data["category"] = spec_info["plant"]

        if state_id == "TX" and crop_year == "weeds_2024" and USDA_symbol in {"URRE2", "URPL2"}:
            log.warning(f'Changing {USDA_symbol} to unknown ("plant") for cutout: {batch_id}/{cutout_id}.json')
            log.critical(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['plant']['class_id']}")
            data["category"] = spec_info["plant"]

        return data

class DateRangeChecker:
    """Class to check and determine crop type based on date ranges."""

    @staticmethod
    def get_crop_type(state: str, datetime_str: str, date_ranges: dict) -> Optional[str]:
        """
        Determine the crop type based on the state and the given date ranges.

        Args:
            state (str): State identifier.
            datetime_str (str): Datetime string.
            date_ranges (dict): Date ranges for crop types.

        Returns:
            Optional[str]: Crop type or None if not found.
        """
        # Convert datetime string to date object
        date_str = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        if state not in date_ranges:
            log.warning(f"State {state} not found in DATE_RANGES.")
            return None

        # Find the crop type based on the date ranges
        for crop_year, periods in date_ranges[state].items():
            start_date = datetime.strptime(periods["start"], "%Y-%m-%d")
            end_date = datetime.strptime(periods["end"], "%Y-%m-%d")
            if start_date <= date_obj <= end_date:
                log.debug(f"Found crop year {crop_year} for date {datetime_str} in state {state}.")
                return crop_year
        
        log.warning(f"No matching crop year found for date {datetime_str} in state {state}.")
        return None

class AnnotationCleaner:
    """Class to clean and reformat individual annotations."""

    def _cutout_exists(cutoutmeta_dir: Path, batch_id: str, cutout_id: str) -> bool:
        """
        Check if a cutout exists for the given batch and cutout ID.

        Args:
            batch_id (str): Batch ID.
            cutout_id (str): Cutout ID.

        Returns:
            bool: True if the cutout exists, False otherwise.
        """
        exists = Path(cutoutmeta_dir, batch_id, cutout_id + ".json").exists()
        log.debug("Cutout exists check for batch_id: %s, cutout_id: %s, exists: %s", batch_id, cutout_id, exists)
        return exists
    
    @staticmethod
    def clean_annotation(annotation: dict, data: dict, category_filepath: Path, date_ranges: dict) -> dict:
        """
        Clean and reformat an annotation by removing unnecessary keys and 
        converting coordinates.

        Args:
            annotation (dict): Annotation dictionary.
            data (dict): Metadata dictionary.
            category_filepath (Path): Path to species JSON file.
            date_ranges (dict): Date ranges for crop types.

        Returns:
            dict: Cleaned and reformatted annotation.
        """
        # Remove unnecessary keys from annotation
        for key in ["local_centroid", "global_centroid", "instance_id", "image_id"]:
            annotation.pop(key, None)

        try:
            # Convert local coordinates to [x, y, width, height]
            annotation["bbox_xywh"] = CoordinateConverter.convert_to_xywh(
                annotation["local_coordinates"], data["width"], data["height"]
            )
        except Exception as e:
            log.error("Error converting coordinates in file: %s", e)

        try:   
            annotation["bbox_area_cm2"] = CoordinateConverter.calculate_bounding_box_area(annotation["global_coordinates"])
            annotation.pop("global_coordinates", None)
        except Exception as e:
            log.error(f"Error calculating bounding box area: {e}")
            return None

        annotation.pop("local_coordinates", None)

        # Reset category based on species info and date ranges
        CategoryManager.reset_category(annotation, category_filepath, data["batch_id"], data["exif_meta"]["DateTime"], date_ranges)
        annotation["category_class_id"] = annotation["category"]["class_id"]
        annotation["cutout_id"] = annotation.pop("bbox_id", None)
        annotation["overlapping_cutout_ids"] = annotation.pop("overlapping_bbox_ids", None)

        if "cutout_exists" not in annotation:
            annotation["cutout_exists"] = AnnotationCleaner._cutout_exists(data["batch_id"], annotation["cutout_id"])

        return annotation

class AnnotationProcessor:
    """Class to process metadata files for image annotations."""

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg (DictConfig): Configuration object with necessary parameters.
        """
        self.cfg = cfg
        self.metadata_dir = Path(cfg.data.batchdir, "metadata")
        self.cutoutmeta_dir = Path(cfg.data.cutoutdir)
        self.category_filepath = cfg.data.species
        self.batch = cfg.general.batch_id
        self.metadata_files = list(self.metadata_dir.glob("*.json"))
        self.date_ranges = cfg.date_ranges

        log.info("AnnotationProcessor initialized with batch ID: %s", self.batch)
        log.info(f"Processing {len(self.metadata_files)}")

    @staticmethod
    def get_unique_categories(dict_list: List[dict], key: str="class_id") -> List[dict]:
        """
        Get unique classes from a list based on their class_id.

        Args:
            dict_list (List[dict]): List of dictionaries.
            key (str): Key to determine uniqueness.

        Returns:
            List[dict]: List of unique dictionaries.
        """
        seen = set()
        unique_dicts = []
        for d in dict_list:
            value = d.get(key)
            if value not in seen:
                unique_dicts.append(d)
                seen.add(value)
        return unique_dicts

    def _cutout_exists(self, batch_id: str, cutout_id: str) -> bool:
        """
        Check if a cutout exists for the given batch and cutout ID.

        Args:
            batch_id (str): Batch ID.
            cutout_id (str): Cutout ID.

        Returns:
            bool: True if the cutout exists, False otherwise.
        """
        exists = Path(self.cutoutmeta_dir, batch_id, cutout_id + ".json").exists()
        log.debug("Cutout exists check for batch_id: %s, cutout_id: %s, exists: %s", batch_id, cutout_id, exists)
        return exists

    def save_metadata(self, meta: Path, data: dict) -> None:
        """
        Save the processed metadata to a file.

        Args:
            meta (Path): Path to the metadata file.
            data (dict): Processed metadata to save.
        """
        try:
            with open(meta, "w") as reformattedfile:
                reformattedfile.write(json.dumps(data, indent=4))
            log.info("Successfully saved file: %s", meta)
        except Exception as e:
            log.error("Failed to save file %s: %s", meta, e)

    def validate_metadata_structure(self, json_data, reference_structure=FULLSIZED_METADATA_SCHEMA) -> bool:
        """
        Validate that the JSON data matches the reference structure exactly, 
        with no extra or missing keys.
        
        Args:
        - json_data (dict): The JSON data to validate.
        - reference_structure (dict): The reference JSON structure.
        
        Returns:
        - bool: True if the structure matches exactly, False otherwise.
        """
        if isinstance(json_data, dict) and isinstance(reference_structure, dict):
            # Check for extra or missing keys
            if set(json_data.keys()) != set(reference_structure.keys()):
                log.debug(f"Key mismatch. JSON keys: {set(json_data.keys())}, Reference keys: {set(reference_structure.keys())}")
                return False
            # Recursively validate each key
            for key in json_data:
                if not self.validate_metadata_structure(json_data[key], reference_structure[key]):
                    return False
        elif isinstance(json_data, list) and isinstance(reference_structure, list) and len(reference_structure) > 0:
            # Validate each item in the list
            for item in json_data:
                if not self.validate_metadata_structure(item, reference_structure[0]):
                    return False
        else:
            # We reached a primitive type; no further validation needed
            return True
        
        return True
    
    def process_file(self, meta: Path) -> None:
        """
        Process a single metadata file by cleaning the metadata, processing annotations,
        and saving the updated metadata.

        Args:
            meta (Path): Path to the metadata file.
        """
        log.info("Processing file: %s", meta)
        try:
            with open(meta, "r") as file:
                metadata = json.load(file)
            metadata = MetadataCleaner.clean_metadata(metadata)
            matches = self.validate_metadata_structure(metadata)
            if matches:
                log.info("Metadata already formatted correctly.")
                return None


            # Convert bboxes to annotations
            metadata["annotations"] = metadata.pop("bboxes")
            annotations = metadata["annotations"]

            categories = []
            reformatted_annotations = []

            for annotation in annotations:
                # Clean and reformat each annotation
                annotation = AnnotationCleaner.clean_annotation(annotation, metadata, self.category_filepath, self.date_ranges)

                if "cutout_exists" not in annotation:
                    annotation["cutout_exists"] = self._cutout_exists(metadata["batch_id"], annotation["cutout_id"])

                reformatted_annotations.append(annotation)
                categories.append(annotation.pop("category", None))

            metadata["annotations"] = reformatted_annotations
            metadata["categories"] = self.get_unique_categories(categories)

            # Save the processed metadata
            self.save_metadata(meta, metadata)
            
        except Exception as e:
            log.exception("Failed to process file %s:", meta)

    def process_all_files_concurrently(self) -> None:
        max_workers = int(len(os.sched_getaffinity(0)) / 5)
        log.info("Processing files concurrently with %d workers", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_file, self.metadata_files)

    def process_all_files_sequentially(self) -> None:
        log.info("Processing files sequentially")
        for metadata in tqdm(self.metadata_files):
            self.process_file(metadata)

def main(cfg: DictConfig) -> None:
    """
    Main function to initialize and run the AnnotationProcessor.

    Args:
        cfg (DictConfig): Configuration object with necessary parameters.
    """
    log.info("Starting AnnotationProcessor.")
    processor = AnnotationProcessor(cfg)
    processor.process_all_files_sequentially()
    log.info("Finished processing files")
