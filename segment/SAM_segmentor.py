import json
import math
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime

import cv2
import torch
import numpy as np
from omegaconf import DictConfig
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from segment_anything_hq import sam_model_registry, SamPredictor

log = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner for performance optimization

class DirectoryInitializer:
    """Class to initialize necessary directories for image processing."""

    def __init__(self, cfg: DictConfig):
        self.batch_id = cfg.general.batch_id
        self.image_dir = Path(cfg.batchdata.images)
        self.metadata_dir = Path(cfg.batchdata.metadata)
        self.cutout_dir = Path(cfg.data.cutoutdir)
        self.meta_mask_dir = Path(cfg.batchdata.meta_masks)

        self._initialize_directories()

    def _initialize_directories(self):
        """Create necessary directories for storing processed data."""
        self.semantic_mask_dir = self.meta_mask_dir / "semantic_masks"
        self.instance_mask_dir = self.meta_mask_dir / "instance_masks"
        self.cutout_batch_dir = self.cutout_dir / self.batch_id

        # Create directories if they do not exist
        for directory in [self.cutout_batch_dir, self.semantic_mask_dir, self.instance_mask_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_images(self) -> List[Path]:
        """Retrieve and return a sorted list of image paths."""
        return sorted(self.image_dir.glob("*.jpg"))


class JSONReadWriter:
    """Static class to load species information from a configuration file."""

    @staticmethod
    def load(read_path: str) -> Dict:
        """
        Load species information from a JSON configuration file.

        Args:
            species_path (str): Path to the species configuration file.

        Returns:
            Dict: Dictionary containing species information.
        """
        log.debug(f"Loading json from {read_path}.")
        with open(read_path, "r") as outfile:
            return json.load(outfile)
        
    @staticmethod
    def save(write_path: str, write_dict: Dict) -> None:
        """
        Load species information from a JSON configuration file.

        Args:
            species_path (str): Path to the species configuration file.

        Returns:
            Dict: Dictionary containing species information.
        """
        log.debug(f"Writing JSON to {write_path}.")
        with open(write_path, "w") as f:
            json.dump(write_dict, f, indent=4, default=str)
            


class BoundingBoxAdjuster:
    """Static class to adjust and fix bounding box categories based on configuration."""

    @staticmethod
    def fix_species_category(cfg: DictConfig, annotation: Dict, batch_id: str, dt: str, cutout_id: int, spec_info: Dict) -> Dict:
        """
        Adjust species category based on state and crop year. Accounts for weed ID or planting mistakes.

        Args:
            cfg (DictConfig): Configuration containing date ranges and batch information.
            annotation (Dict): Annotation dictionary containing category and bounding box info.
            batch_id (str): Batch identifier.
            dt (str): Date and time as a string.
            cutout_id (int): Cutout identifier.
            spec_info (Dict): Species information dictionary.

        Returns:
            Dict: Adjusted species information.
        """
        class_id = annotation["category_class_id"]
        annot_cat = [spec_info[d] for d in spec_info if spec_info[d]["class_id"] == class_id][0]
        
        USDA_symbol = annot_cat["USDA_symbol"]
        date_ranges = cfg.date_ranges
        batch_id = cfg.general.batch_id
        state_id = batch_id.split("_")[0]
        
        crop_year = BoundingBoxAdjuster.get_crop_year(date_ranges, state_id, dt)
        
        # Handle specific cases based on state and crop year
        if state_id == "MD" and crop_year == "weeds 2023":
            if USDA_symbol in ["ELIN3", "URPL2"]:
                log.warning(f'Changing {USDA_symbol} to unknown ("plant") for cutout: {batch_id}/{cutout_id}.json')
                log.warning(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['plant']['class_id']}")
                return spec_info["plant"]

        elif state_id == "NC" and crop_year == "weeds 2022":
            if USDA_symbol == "URPL2":
                log.warning(f'Changing {USDA_symbol} to Texas Millet ("URTE2") for cutout: {batch_id}/{cutout_id}.json')
                log.warning(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['URTE2']['class_id']}")
                return spec_info["URTE2"]

        elif state_id == "TX" and crop_year in ["weeds 2023", "weeds 2024"]:
            if USDA_symbol in ["ECCO2", "URRE2", "URPL2"]:
                log.warning(f'Changing {USDA_symbol} to unknown ("plant") for cutout: {batch_id}/{cutout_id}.json')
                log.warning(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['plant']['class_id']}")
                return spec_info["plant"]
        
        return spec_info[USDA_symbol]

    @staticmethod
    def get_crop_year(date_ranges: Dict, state: str, datetime_str: str) -> Optional[str]:
        """
        Determine crop year based on date and state.

        Args:
            date_ranges (Dict): Date ranges for crops in different states.
            state (str): State identifier.
            datetime_str (str): Date and time string.

        Returns:
            Optional[str]: Crop year if found, otherwise None.
        """
        date_str = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if state not in date_ranges:
            log.warning(f"State {state} not found in DATE_RANGES.")
            return None

        for crop_year, periods in date_ranges[state].items():
            start_date = datetime.strptime(periods["start"], "%Y-%m-%d")
            end_date = datetime.strptime(periods["end"], "%Y-%m-%d")
            if start_date <= date_obj <= end_date:
                log.debug(f"Found crop year {crop_year} for date {datetime_str} in state {state}.")
                return crop_year
        log.warning(f"No matching crop year found for date {datetime_str} in state {state}.")
        return None


class MaskCreator:
    """Class to create masks using SAM predictor."""

    def __init__(self, mask_predictor: SamPredictor):
        """
        Initialize MaskCreator with a SAM predictor.

        Args:
            mask_predictor (SamPredictor): SAM predictor instance.
        """
        self.mask_predictor = mask_predictor

    def create_masks(self, image: np.ndarray, annotations: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create class and instance masks for an image based on annotations.

        Args:
            image (np.ndarray): Image array.
            annotations (List[Dict]): List of annotation dictionaries.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Class mask, instance mask, and updated annotations.
        """
        im_size_X, im_size_Y = image.shape[1], image.shape[0]
        im_pad_size = 1500
        
        # Expand image to avoid edge issues during mask creation
        image_expanded = cv2.copyMakeBorder(image, im_pad_size, im_pad_size, im_pad_size, im_pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        masked_image = np.copy(image_expanded)
        class_masked_image = np.zeros(masked_image.shape[0:2])
        instance_masked_image = np.zeros(masked_image.shape[0:2], dtype=np.uint16)

        self.update_annotations_with_absolute_coords(annotations)

        # Sort annotations by area in descending order
        annotations.sort(key=lambda ann: ann['bbox_xywh'][2] * ann['bbox_xywh'][3], reverse=True)

        for index, annotation in enumerate(annotations):
            self.process_annotation(annotation, image_expanded, class_masked_image, instance_masked_image, im_pad_size, index, im_size_X, im_size_Y)

        # Remove padding from masks
        return class_masked_image[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size], instance_masked_image[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size], annotations

    def update_annotations_with_absolute_coords(self, annotations: List[Dict]):
        """
        Update annotations with absolute coordinates.

        Args:
            annotations (List[Dict]): List of annotation dictionaries.
        """
        for annotation in annotations:
            annotation['centerX'] = (annotation['bbox_xywh'][0] + annotation['bbox_xywh'][2] / 2)
            annotation['centerY'] = (annotation['bbox_xywh'][1] + annotation['bbox_xywh'][3] / 2)

    def process_annotation(self, annotation: Dict, image_expanded: np.ndarray,
                           class_masked_image: np.ndarray, instance_masked_image: np.ndarray,
                           im_pad_size: int, index: int, im_size_X: int, im_size_Y: int):
        """
        Process individual annotation to create masks.

        Args:
            annotation (Dict): Annotation dictionary.
            image_expanded (np.ndarray): Expanded image array.
            class_masked_image (np.ndarray): Class masked image.
            instance_masked_image (np.ndarray): Instance masked image.
            im_pad_size (int): Padding size.
            index (int): Annotation index.
            im_size_X (int): Image width.
            im_size_Y (int): Image height.
        """
        minX, minY, width, height = annotation['bbox_xywh']
        maxX = minX + width
        maxY = minY + height
        plant_bbox = np.array([minX, minY, maxX, maxY])

        sam_crop_size_x, sam_crop_size_y = self.determine_crop_size(width, height)

        cropped_image = self.crop_image(image_expanded, annotation, im_pad_size, sam_crop_size_x, sam_crop_size_y)

        self.mask_predictor.set_image(cropped_image)
        log.info(f"Cropped image size for SAM predictor: {cropped_image.shape}")

        _, cropped_bbox = self.get_bounding_boxes(annotation, plant_bbox, im_pad_size, sam_crop_size_x, sam_crop_size_y)
        input_boxes = torch.tensor(cropped_bbox, device=self.mask_predictor.device)

        transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(input_boxes, cropped_image.shape[:2])

        masks, _, _ = self.mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes,
                                                        multimask_output=True, hq_token_only=False)

        self.apply_masks(masks, class_masked_image, instance_masked_image, annotation, im_pad_size, sam_crop_size_x, sam_crop_size_y, index)

    def determine_crop_size(self, width: float, height: float) -> Tuple[int, int]:
        """
        Determine the crop size based on bounding box dimensions.

        Args:
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.

        Returns:
            Tuple[int, int]: Crop size for SAM predictor.
        """
        sam_crop_size_x = 1000
        sam_crop_size_y = 1000
        if width > 700:
            sam_crop_size_x = math.ceil(width * 1.43 / 2.) * 2
        if height > 700:
            sam_crop_size_y = math.ceil(height * 1.43 / 2.) * 2
        return sam_crop_size_x, sam_crop_size_y

    def crop_image(self, image_expanded: np.ndarray, annotation: Dict, im_pad_size: int,
                   sam_crop_size_x: int, sam_crop_size_y: int) -> np.ndarray:
        """
        Crop the image based on annotation center and crop size.

        Args:
            image_expanded (np.ndarray): Expanded image array.
            annotation (Dict): Annotation dictionary.
            im_pad_size (int): Padding size.
            sam_crop_size_x (int): Crop width.
            sam_crop_size_y (int): Crop height.

        Returns:
            np.ndarray: Cropped image array.
        """
        centerX = annotation['centerX']
        centerY = annotation['centerY']
        return np.copy(image_expanded[int(centerY + im_pad_size - sam_crop_size_y / 2):int(centerY + im_pad_size + sam_crop_size_y / 2),
                                      int(centerX + im_pad_size - sam_crop_size_x / 2):int(centerX + im_pad_size + sam_crop_size_x / 2), :])

    def get_bounding_boxes(self, annotation: Dict, plant_bbox: np.ndarray, im_pad_size: int,
                           sam_crop_size_x: int, sam_crop_size_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get padded and cropped bounding boxes for an annotation.

        Args:
            annotation (Dict): Annotation dictionary.
            plant_bbox (np.ndarray): Bounding box array.
            im_pad_size (int): Padding size.
            sam_crop_size_x (int): Crop width.
            sam_crop_size_y (int): Crop height.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Padded and cropped bounding boxes.
        """
        centerX = annotation['centerX']
        centerY = annotation['centerY']
        padded_bbox = plant_bbox + [im_pad_size, im_pad_size, im_pad_size, im_pad_size]
        cropped_bbox = padded_bbox - [centerX + im_pad_size - sam_crop_size_x / 2,
                                      centerY + im_pad_size - sam_crop_size_y / 2,
                                      centerX + im_pad_size - sam_crop_size_x / 2,
                                      centerY + im_pad_size - sam_crop_size_y / 2]
        return padded_bbox, cropped_bbox

    def apply_masks(self, masks: torch.Tensor, class_masked_image: np.ndarray,
                    instance_masked_image: np.ndarray, annotation: Dict, im_pad_size: int,
                    sam_crop_size_x: int, sam_crop_size_y: int, index: int):
        """
        Apply masks to class and instance masked images.

        Args:
            masks (torch.Tensor): Tensor of predicted masks.
            class_masked_image (np.ndarray): Class masked image.
            instance_masked_image (np.ndarray): Instance masked image.
            annotation (Dict): Annotation dictionary.
            im_pad_size (int): Padding size.
            sam_crop_size_x (int): Crop width.
            sam_crop_size_y (int): Crop height.
            index (int): Annotation index.
        """
        centerX = annotation['centerX']
        centerY = annotation['centerY']
        crop_start_y = int(centerY + im_pad_size - sam_crop_size_y / 2)
        crop_end_y = int(centerY + im_pad_size + sam_crop_size_y / 2)
        crop_start_x = int(centerX + im_pad_size - sam_crop_size_x / 2)
        crop_end_x = int(centerX + im_pad_size + sam_crop_size_x / 2)

        for mask in masks:
            full_mask = np.zeros(class_masked_image.shape[0:2])
            # Directly get the mask data from the tensor
            mask_data = mask.cpu()[0, :, :]
            
            # Update the relevant slice of full_mask
            full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = mask_data
            # Vectorized updates for class_masked_image and instance_masked_image
            class_id = annotation['category_class_id']
            class_value = class_id if class_id != 28 else 0
            class_masked_image[full_mask == 1] = class_value
            instance_masked_image[full_mask == 1] = index + 1 if class_id != 28 else 0



class ImageProcessor:
    """Class to handle the processing of images and annotations."""

    def __init__(self, cfg: DictConfig, directory_initializer: DirectoryInitializer, species_info: Dict, device: str = "cuda"):
        """
        Initialize ImageProcessor with configuration and directories.

        Args:
            cfg (DictConfig): Configuration containing processing parameters.
            directory_initializer (DirectoryInitializer): Initialized directories.
            species_info (Dict): Species information dictionary.
            device (str): Device to use for processing (default is "cuda").
        """
        log.info("Initializing ImageProcessor")
        self.directory_initializer = directory_initializer
        self.instance_label_dir = directory_initializer.instance_mask_dir
        self.class_label_dir = directory_initializer.semantic_mask_dir

        self.species_info = species_info
        self.cfg = cfg

        log.info("Loading SAM model")
        sam = sam_model_registry[cfg.segment.sam.model_type](checkpoint=cfg.segment.sam.sam_checkpoint)
        sam.to(device=device)
        self.mask_creator = MaskCreator(SamPredictor(sam))

    def preprocess_single_image_annotations(self, annotations: Dict, data: Dict) -> List[Dict]:
        """
        Preprocess annotations to fix category class IDs.

        Args:
            annotations (Dict): Dictionary of annotations.
            dt (datetime): datetime object from image exif

        Returns:
            List[Dict]: List of preprocessed annotations.
        """
        batch_id = self.cfg.general.batch_id
        dt = data["exif_meta"]["DateTime"]
        image_id = data["image_id"]
        for idx, annotation in enumerate(annotations):
            annotation["image_id"] = image_id
            annotation["cutout_id"] = f"{image_id}_{idx}"
            new_cat_id = BoundingBoxAdjuster.fix_species_category(self.cfg, annotation, batch_id, dt, idx, self.species_info)
            annotation["category_class_id"] = new_cat_id["class_id"]
    
        return annotations
    
    def preprocess_annotations(self, annotations: Dict) -> List[Dict]:
        """
        Preprocess annotations to fix category class IDs and add cutout IDs.

        Args:
            annotations (Dict): Dictionary containing annotations for each image.

        Returns:
            List[Dict]: List of processed annotations with updated category class IDs and cutout IDs.
        """
        batch_id = self.cfg.general.batch_id
        
        for idx, annotation in enumerate(annotations):
            dt = annotation["DateTime"]
            image_id = annotation["image_id"]
            annotation["cutout_id"] = f"{image_id}_{idx}"
            new_cat_id = BoundingBoxAdjuster.fix_species_category(self.cfg, annotation, batch_id, dt, idx, self.species_info)
            annotation["category_class_id"] = new_cat_id["class_id"]
    
        return annotations

    def post_process_metadata(self, metadata, annotations):
        """
        Update metadata with processed annotations and match categories.

        Args:
            metadata (Dict): Existing metadata dictionary.
            annotations (List[Dict]): List of processed annotations.

        Returns:
            Dict: Updated metadata with annotations and categories.
        """
        # Create a mapping of new annotations using bbox_xywh for quick lookup
        new_annotations_map = {tuple(ann['bbox_xywh']): ann for ann in annotations}
        
        # Counter for updated and unmatched annotations
        updated_count = 0
        unmatched_count = 0

        # Update old annotations with data from new annotations
        for old_ann in metadata["annotations"]:
            bbox = tuple(old_ann['bbox_xywh'])
            if bbox in new_annotations_map:
                new_ann = new_annotations_map[bbox]
                # Update fields as needed
                old_ann['category_class_id'] = new_ann['category_class_id']
                old_ann['cutout_id'] = new_ann['cutout_id']
                updated_count += 1
            else:
                unmatched_count += 1
                log.critical(f"No match found for bbox {bbox}")

        # Logging summary
        log.info(f"Total annotations updated: {updated_count}")
        log.info(f"Total unmatched annotations: {unmatched_count}")
        
        unique_category_ids = set({item['category_class_id'] for item in annotations})

        # Find categories that match the unique category IDs
        unique_category_dicts = [
            category_dict 
            for annot_category_id in unique_category_ids
            for category_dict in self.species_info.values()
            if category_dict["class_id"] == annot_category_id
            ]

        metadata["categories"] = unique_category_dicts
        return metadata

    def save_metadata(self, annotations):
        """
        Save processed metadata and annotations to a new JSON file.

        Args:
            annotations (List[Dict]): List of processed annotations.
        """
        image_id = annotations[0]["image_id"]
        metadata_path = Path(self.directory_initializer.metadata_dir, image_id + ".json")
        metadata = JSONReadWriter.load(metadata_path)

        metadata = self.post_process_metadata(metadata, annotations)

        write_path = Path(self.directory_initializer.metadata_dir, image_id + ".json")
        log.info(f"Saving metadata to {write_path}.")
        JSONReadWriter.save(write_path, metadata)
        
        
    def process_single_image(self, input_paths: Tuple[Path, Path]):
        """
        Process a single image and generate masks.

        Args:
            input_paths (Tuple[Path, Path]): Tuple containing image and JSON paths.
        """
        image_path, json_path = input_paths
        log.info(f"Processing image: {image_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        annotations = data['annotations']
        if not annotations:
            log.warning(f"No annotations found for image {image_path}")
            return
        
        log.info(f"Preprocessing annotations ({len(annotations)}).")
        annotations = self.preprocess_single_image_annotations(annotations, data)

        image = self.read_image(image_path)

        class_masked_image, instance_masked_image, annotations = self.mask_creator.create_masks(image, annotations)
        
        log.info(f"Saving metadata...")
        self.save_metadata(annotations)

        mask_name = Path(image_path).stem + '.png'
        log.info(f"Saving masks ({Path(image_path).stem}) to {self.instance_label_dir.parent}")
        cv2.imwrite(str(self.instance_label_dir / mask_name), instance_masked_image.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(self.class_label_dir / mask_name), class_masked_image.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])

    @staticmethod
    def read_image(image_path: Path) -> np.ndarray:
        """
        Read and convert an image to RGB format.

        Args:
            image_path (Path): Path to the image.

        Returns:
            np.ndarray: Image array in RGB format.
        """
        log.info(f"Reading image and converting to RGB: {image_path}")
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)


class ImageDataset(Dataset):
    """Dataset class for handling image and annotation data."""

    def __init__(self, data_paths: List[Tuple[Path, Path]], metadata_dir: Path, max_annotations: int = 2000):
        """
        Initialize ImageDataset with data paths and metadata.

        Args:
            data_paths (List[Tuple[Path, Path]]): List of tuples containing image and JSON paths.
            metadata_dir (Path): Directory containing metadata.
            max_annotations (int): Maximum number of annotations (default is 2000).
        """
        self.data_paths = data_paths
        self.metadata_dir = metadata_dir
        self.max_annotations = max_annotations

    def __len__(self) -> int:
        """Return the number of data paths."""
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Dict], str]:
        """
        Get an item from the dataset at a specific index.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, List[Dict], str]: Image tensor, padded annotations, and image stem.
        """
        data_path = self.data_paths[idx]
        image_path = Path(data_path[0])
        json_path = Path(data_path[1])
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

        # Convert image to tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW

        with open(json_path, 'r') as f:
            metadata = json.load(f)
            annotations = metadata['annotations']
            for annotation in annotations:
                annotation["image_id"] = metadata["image_id"]
                annotation["batch_id"] = metadata["batch_id"]
                annotation["DateTime"] = metadata["exif_meta"]["DateTime"]
                
            # Convert annotations to a fixed-size tensor
            padded_annotations = self._annotations_padded(annotations)
            
        # Log shapes and types for debugging
        return image_tensor, padded_annotations, image_path.stem
    
    def _annotations_padded(self, annotations: List[Dict]) -> List[Dict]:
        """
        Pad or truncate annotations to a fixed size.

        Args:
            annotations (List[Dict]): List of annotation dictionaries.

        Returns:
            List[Dict]: Padded or truncated annotations.
        """
        annotations_list = []
        for annotation in annotations:
            # Extract numerical and string components
            bbox = annotation.get('bbox_xywh', [0, 0, 0, 0])
            class_id = annotation.get('category_class_id', -1)
            image_id = annotation.get('image_id', '')
            batch_id = annotation.get('batch_id', '')
            date_time = annotation.get('DateTime', '')

            # Store as a dictionary
            annotations_list.append({
                'bbox_xywh': bbox,
                'category_class_id': class_id,
                'image_id': image_id,
                'batch_id': batch_id,
                'DateTime': date_time,
                
            })

        # Pad or truncate the annotations to `max_annotations` size
        padded_annotations = annotations_list[:self.max_annotations]
        while len(padded_annotations) < self.max_annotations:
            padded_annotations.append({
                'bbox_xywh': [0, 0, 0, 0],
                'category_class_id': -1,
                'image_id': '',
                'batch_id': '',
                'DateTime': ''
            })
        return padded_annotations


def separate_dictionaries(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Separate a list of dictionaries into individual dictionaries based on image IDs.

    Args:
        data (List[Dict[str, Any]]): List of dictionaries containing image data.

    Returns:
        List[Dict[str, Any]]: List of separated dictionaries.
    """
    separated_data = []
    
    for entry in data:
        num_images = len(entry['image_id'])
        
        # Iterate over each image_id and create a separate dictionary
        for i in range(num_images):
            individual_dict = {}
            
            # Assign values for each image_id
            individual_dict['image_id'] = entry['image_id'][i]
            individual_dict['DateTime'] = entry['DateTime'][i]
            individual_dict['batch_id'] = entry['batch_id'][i]
            
            # Extract bbox_xywh values for each image
            individual_dict['bbox_xywh'] = [
                entry['bbox_xywh'][0][i].item(), 
                entry['bbox_xywh'][1][i].item(), 
                entry['bbox_xywh'][2][i].item(), 
                entry['bbox_xywh'][3][i].item()
            ]
            
            # Convert category_class_id tensor to int
            individual_dict['category_class_id'] = int(entry['category_class_id'][i])
         
            separated_data.append(individual_dict)
    
    return separated_data


def group_annotations_by_image(images: List[np.ndarray], annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group annotations by corresponding image.

    Args:
        images (List[np.ndarray]): List of image arrays.
        annotations (List[Dict[str, Any]]): List of annotation dictionaries.

    Returns:
        List[Dict[str, Any]]: List of grouped data containing images and annotations.
    """
    grouped_data = [{'image': image, 'annotations': []} for image in images]
    
    for annotation in annotations:
        image_id = annotation['image_id']
        
        for index, data in enumerate(grouped_data):
            if data['annotations'] and data['annotations'][0]['image_id'] == image_id:
                data['annotations'].append(annotation)
                break
            elif not data['annotations']:
                data['annotations'].append(annotation)
                break
    
    return grouped_data

def process_batch(batch: Tuple[torch.Tensor, List[Dict], str], processor: ImageProcessor):
    """
    Process a batch of images and annotations using the ImageProcessor.

    Args:
        batch (Tuple[torch.Tensor, List[Dict], str]): Batch containing image tensor, annotations, and image stem.
        processor (ImageProcessor): Instance of ImageProcessor.
    """
    images, annotations_list, _ = batch
    separated_annotations = separate_dictionaries(annotations_list)
    re_orged_batches = group_annotations_by_image(images, separated_annotations)

    for data in re_orged_batches:
        annotations = data["annotations"]
        stem = annotations[0]["image_id"]

        image_tensor = data["image"]

        # Convert image tensor back to numpy array for processing
        image_np = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)  # Convert to HxWxC

        log.info(f"Number of annotations before preprocessing for {stem}: {len(annotations)}")

        annotations = processor.preprocess_annotations(annotations)
        log.info(f"Number of annotations after preprocessing for {stem}: {len(annotations)}")

        # If there are no valid annotations, skip processing
        if not annotations:
            log.warning(f"No valid annotations found for image {stem}.")
            continue

        # Generate masks using the MaskCreator
        class_masked_image, instance_masked_image, annotations = processor.mask_creator.create_masks(image_np, annotations)

        log.info(f"Saving metadata...")
        processor.save_metadata(annotations)
        
        # Save the generated masks
        mask_name = f"{stem}.png"
        log.info(f"Saving masks ({Path(mask_name)}) to {processor.instance_label_dir.parent}")
        cv2.imwrite(str(processor.instance_label_dir / mask_name), instance_masked_image.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(processor.class_label_dir / mask_name), class_masked_image.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def worker(gpu_id: int, input_paths: List[Tuple[Path, Path]], cfg: DictConfig, species_info: Dict, directory_initializer: DirectoryInitializer):
    """
    Worker function for processing images on a specific GPU.

    Args:
        gpu_id (int): GPU identifier.
        input_paths (List[Tuple[Path, Path]]): List of input paths containing image and JSON paths.
        cfg (DictConfig): Configuration containing processing parameters.
        species_info (Dict): Species information dictionary.
        directory_initializer (DirectoryInitializer): Initialized directories.
    """
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    log.info(f"Process on GPU {gpu_id} started.")
    processor = ImageProcessor(cfg, directory_initializer, species_info, device=device)

    dataset = ImageDataset(input_paths, directory_initializer.metadata_dir)
    batch_size = cfg.segment.sam.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in dataloader:
        process_batch(batch, processor)


def process_with_torch_multiprocessing(cfg: DictConfig):
    """
    Process images using torch multiprocessing on multiple GPUs.

    Args:
        cfg (DictConfig): Configuration containing processing parameters.
    """
    directory_initializer = DirectoryInitializer(cfg)
    species_info = JSONReadWriter.load(cfg.data.species)["species"]
    imgs = directory_initializer.get_images()
    input_paths = [(image_path, str(directory_initializer.metadata_dir / f"{image_path.stem}.json")) for image_path in imgs]

    # Split the input paths into chunks for each GPU
    num_gpus = cfg.segment.sam.number_gpus
    chunk_size = len(input_paths) // num_gpus
    chunks = [input_paths[i:i + chunk_size] for i in range(0, len(input_paths), chunk_size)]
    
    log.info(f"Chunk size: {chunk_size}")
    log.info(f"Chunk length: {len(chunks)}")

    # Launch multiple processes using torch.multiprocessing
    processes = []
    for gpu_id in range(num_gpus):
        log.info(f"Image chunk length: {len(chunks[gpu_id])}")
        p = mp.Process(target=worker, args=(gpu_id, chunks[gpu_id], cfg, species_info, directory_initializer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    log.info("Finished processing with torch multiprocessing.")


def process_sequentially(cfg: DictConfig):
    """
    Process images sequentially without multiprocessing.

    Args:
        cfg (DictConfig): Configuration containing processing parameters.
    """
    directory_initializer = DirectoryInitializer(cfg)
    species_info = JSONReadWriter.load(cfg.data.species)["species"]
    processor = ImageProcessor(cfg, directory_initializer, species_info)
    imgs = directory_initializer.get_images()

    for image_path in imgs:
        log.info(f"Processing image: {image_path}")
        json_path = str(directory_initializer.metadata_dir / f"{image_path.stem}.json")
        input_paths = (image_path, json_path)
        processor.process_single_image(input_paths)
    log.info("Finished processing sequentially.")


def main(cfg: DictConfig):
    """
    Main function to choose processing method based on configuration.

    Args:
        cfg (DictConfig): Configuration containing processing parameters.
    """
    if cfg.segment.sam.multiprocess:
        process_with_torch_multiprocessing(cfg)
    else:
        process_sequentially(cfg)
