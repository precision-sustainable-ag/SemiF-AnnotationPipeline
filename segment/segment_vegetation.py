import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from omegaconf import DictConfig
from scipy.stats import zscore
from tqdm import tqdm

from semif_utils.datasets import Cutout, ImageData
from semif_utils.segment_species import Segment
from semif_utils.segment_utils import GenCutoutProps, generate_new_color, prep_bbox
from semif_utils.utils import apply_mask, create_dataclasses, cutoutmeta2csv, reduce_holes

log = logging.getLogger(__name__)

class DirectoryInitializer:
    def __init__(self, cfg: DictConfig):
        """
        Initialize directory structure based on the configuration.

        Args:
            cfg (DictConfig): Configuration object containing directory paths and other settings.
        """
        self.cfg = cfg
        self.cutout_dir = Path(cfg.data.cutoutdir)
        self.season = cfg.general.season
        self.batch_id = Path(cfg.data.batchdir).name
        self.metadata = cfg.batchdata.metadata
        self.multi_process = cfg.segment.multiprocess

        self._initialize_directories()

    def _initialize_directories(self):
        """Create necessary directories for cutout and mask storage."""
        self.binary_mask_dir = Path(self.cfg.batchdata.meta_masks, "binary_masks")
        self.semantic_mask_dir = Path(self.cfg.batchdata.meta_masks, "semantic_masks")
        self.instance_mask_dir = Path(self.cfg.batchdata.meta_masks, "instance_masks")
        self.cutout_batch_dir = self.cutout_dir / self.batch_id

        for directory in [self.cutout_batch_dir, self.semantic_mask_dir, self.instance_mask_dir]:
            directory.mkdir(parents=True, exist_ok=True)

class BBoxStatsCalculator:
    def __init__(self, cfg: DictConfig):
        """
        Calculate bounding box statistics and outlier thresholds.

        Args:
            cfg (DictConfig): Configuration object containing dataset and processing settings.
        """
        self.cfg = cfg
        self.bbox_stats = self.calculate_bbox_stats_per_species(create_dataclasses(Path(f"{cfg.data.batchdir}/metadata"), cfg))
        self.outlier_thresholds = self.calculate_outlier_thresholds()

    def calculate_bbox_stats_per_species(self, image_data_list: List) -> Dict:
        """
        Calculate bounding box area statistics per species.

        Args:
            image_data_list (List): List of image data objects containing bounding box information.

        Returns:
            Dict: Dictionary with species names as keys and statistics as values.
        """
        bbox_areas_per_species = defaultdict(list)
        for img_data in image_data_list:
            scale = [img_data.fullres_width, img_data.fullres_height]
            for box in img_data.bboxes:
                # Prepare the bounding box coordinates
                _, x1, y1, x2, y2 = prep_bbox(box, scale, save_changes=False)
                width = float(x2) - float(x1)
                length = float(y2) - float(y1)
                area = width * length
                bbox_areas_per_species[box.cls["common_name"]].append(area)

        bbox_stats = {
            species: {
                "areas": areas,
                "mean": np.mean(areas),
                "25th_percentile": np.percentile(areas, 25)
            }
            for species, areas in bbox_areas_per_species.items()
        }
        return bbox_stats

    def calculate_outlier_thresholds(self) -> Dict:
        """
        Calculate outlier thresholds for bounding box areas.

        Returns:
            Dict: Dictionary with species names as keys and outlier thresholds as values.
        """
        outlier_thresholds = {}
        for species, stats in self.bbox_stats.items():
            areas = np.array(stats["areas"])
            z_scores = zscore(areas)
            threshold = 2.0  # Z-score threshold for defining outliers
            lower_mask = z_scores < -threshold
            upper_mask = z_scores > threshold

            lower_bound = np.min(areas[~lower_mask]) if (~lower_mask).any() else 0  
            upper_bound = np.max(areas[~upper_mask]) if (~upper_mask).any() else np.inf

            outlier_thresholds[species] = (float(lower_bound), float(upper_bound))
        return outlier_thresholds

class CutoutProcessor:
    def __init__(self, cfg: DictConfig, bbox_stats_calculator: BBoxStatsCalculator, dir_initializer: DirectoryInitializer):
        """
        Process cutouts based on bounding box statistics and configuration.

        Args:
            cfg (DictConfig): Configuration object containing dataset and processing settings.
            bbox_stats_calculator (BBoxStatsCalculator): Calculator object for bounding box statistics.
            dir_initializer (DirectoryInitializer): Initializer object for directory setup.
        """
        self.cfg = cfg
        self.season = cfg.general.season
        self.batch_id = Path(cfg.data.batchdir).name
        self.species_path = cfg.data.species
        self.date_ranges = cfg.date_ranges
        self.bbox_stats_calculator = bbox_stats_calculator
        self.cutout_dir = dir_initializer.cutout_dir
        self.semantic_mask_dir = dir_initializer.semantic_mask_dir
        self.instance_mask_dir = dir_initializer.instance_mask_dir
        self.metadata = cfg.batchdata.metadata

    def config_from_bboxarea(self, boxarea: float) -> Tuple[int, int, int]:
        """
        Return configuration parameters based on bounding box area.

        Args:
            boxarea (float): Area of the bounding box.

        Returns:
            Tuple[int, int, int]: Configuration parameters.
        """
        if boxarea < 25000:
            return 100, 100, 1
        elif boxarea < 50000:
            return 500, 500, 3
        elif boxarea < 100000:
            return 1000, 1000, 7
        elif boxarea < 200000:
            return 5000, 5000, 9
        else:
            return 10000, 10000, 11

    def create_instance_mask(self, instance_mask: np.ndarray, instance_palette: List, box_instance_id: int) -> np.ndarray:
        """
        Create an instance mask from the provided palette and instance ID.

        Args:
            instance_mask (np.ndarray): Array representing the instance mask.
            instance_palette (List): List of unique instance IDs.
            box_instance_id (int): ID of the current instance.

        Returns:
            np.ndarray: Color mask for the instance.
        """
        color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
        for label_id in np.unique(instance_palette):
            if label_id == 0:
                continue
            if label_id != box_instance_id:
                color_mask[instance_mask == label_id] = box_instance_id
        return color_mask

    def create_semantic_mask(self, semantic_mask: np.ndarray, box_instance_id: int) -> np.ndarray:
        """
        Create a semantic mask with the provided instance ID.

        Args:
            semantic_mask (np.ndarray): Array representing the semantic mask.
            box_instance_id (int): ID of the current instance.

        Returns:
            np.ndarray: Modified semantic mask.
        """
        semantic_mask[semantic_mask == 255] = box_instance_id
        return semantic_mask

    def load_species_info(self) -> dict:
        """
        Load species information from the configuration file.

        Returns:
            dict: Dictionary containing species information.
        """
        log.debug(f"Loading species info from {self.species_path}.")
        with open(self.species_path, "r") as outfile:
            return json.load(outfile)

    def get_crop_type(self, state: str, datetime_str: str) -> Optional[str]:
        """
        Get crop type based on state and datetime string.

        Args:
            state (str): State identifier.
            datetime_str (str): Datetime string in the format "%Y:%m:%d %H:%M:%S".

        Returns:
            Optional[str]: Crop type if found, otherwise None.
        """
        date_str = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if state not in self.date_ranges:
            log.warning(f"State {state} not found in DATE_RANGES.")
            return None

        for crop_year, periods in self.date_ranges[state].items():
            start_date = datetime.strptime(periods["start"], "%Y-%m-%d")
            end_date = datetime.strptime(periods["end"], "%Y-%m-%d")
            if start_date <= date_obj <= end_date:
                log.debug(f"Found crop year {crop_year} for date {datetime_str} in state {state}.")
                return crop_year
        log.warning(f"No matching crop year found for date {datetime_str} in state {state}.")
        return None

    def fix_category(self, batch_id: str, category: dict, cutout_id: str, dt: str) -> dict:
        """
        Fix category based on batch ID, category data, cutout ID, and datetime.

        Args:
            batch_id (str): Batch identifier.
            category (dict): Category information.
            cutout_id (str): Cutout identifier.
            dt (str): Datetime string.

        Returns:
            dict: Updated category information.
        """
        spec_info = self.load_species_info()["species"]
        state_id = batch_id.split("_")[0]
        USDA_symbol = category["USDA_symbol"]
        class_id = category["class_id"]

        crop_year = self.get_crop_type(state_id, dt)

        # Handle specific cases based on state and crop year
        if state_id == "MD" and crop_year == "weeds_2023":
            if USDA_symbol in ["ELIN3", "URPL2"]:
                log.warning(f'Changing {USDA_symbol} to unknown ("plant") for cutout: {batch_id}/{cutout_id}.json')
                log.critical(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['plant']['class_id']}")
                return spec_info["plant"]

        elif state_id == "NC" and crop_year == "weeds_2022":
            if USDA_symbol == "URPL2":
                log.warning(f'Changing {USDA_symbol} to Texas Millet ("URTE2") for cutout: {batch_id}/{cutout_id}.json')
                log.critical(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['URTE2']['class_id']}")
                return spec_info["URTE2"]

        elif state_id == "TX" and crop_year in ["weeds_2023", "weeds_2024"]:
            if USDA_symbol in ["ECCO2", "URRE2", "URPL2"]:
                log.warning(f'Changing {USDA_symbol} to unknown ("plant") for cutout: {batch_id}/{cutout_id}.json')
                log.critical(f"Remap mask {batch_id}/{cutout_id}.json {class_id} to {spec_info['plant']['class_id']}")
                return spec_info["plant"]

        return spec_info[USDA_symbol]

    def process_bbox(self, imgdata: ImageData, rgb_array: np.ndarray, bboxes: List, instance_colors: List, instance_palette: List, semantic_palette: List) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Process bounding boxes to generate cutouts and masks.

        Args:
            imgdata (ImageData): Image dataclass object containing bounding boxes.
            rgb_array (np.ndarray): Array representing the RGB image.
            bboxes (List): List of bounding boxes.
            instance_colors (List): List of instance colors.
            instance_palette (List): List of instance palette colors.
            semantic_palette (List): List of semantic palette colors.

        Returns:
            Tuple[List, np.ndarray, np.ndarray]: List of cutout IDs, semantic mask, and instance mask.
        """
        cutout_ids = []
        semantic_mask_zeros = np.zeros(rgb_array.shape[:2], dtype=np.float32)
        instance_mask_zeros = np.zeros(rgb_array.shape, dtype=np.float32)

        for box in bboxes:
            self.process_single_box(imgdata, rgb_array, box, instance_colors, instance_palette, semantic_palette, cutout_ids, semantic_mask_zeros, instance_mask_zeros)

        return cutout_ids, semantic_mask_zeros, instance_mask_zeros

    def process_single_box(self, imgdata: ImageData, rgb_array: np.ndarray, box: dict, instance_colors: List, instance_palette: List, semantic_palette: List, cutout_ids: List, semantic_mask_zeros: np.ndarray, instance_mask_zeros: np.ndarray):
        """
        Process a single bounding box to create cutouts and masks.

        Args:
            imgdata (ImageData): Image dataclass object containing bounding boxes.
            rgb_array (np.ndarray): Array representing the RGB image.
            box (dict): Dictionary representing a single bounding box.
            instance_colors (List): List of instance colors.
            instance_palette (List): List of instance palette colors.
            semantic_palette (List): List of semantic palette colors.
            cutout_ids (List): List to store cutout IDs.
            semantic_mask_zeros (np.ndarray): Array representing the semantic mask.
            instance_mask_zeros (np.ndarray): Array representing the instance mask.
        """
        # Scale the bounding box coordinates
        x1, y1, x2, y2 = self.scale_bbox(imgdata, box)
        # Get the cropped RGB array
        rgb_crop = self.get_rgb_crop_and_data(rgb_array, x1, y1, x2, y2)
        seg = Segment(rgb_crop, species=box.cls, bbox=(x1, y1, x2, y2))
        boxarea = seg.get_bboxarea()

        if not self.is_valid_boxarea(boxarea, box, seg, rgb_crop, (x1, y1, x2, y2)):
            return

        # Get configuration parameters based on the bounding box area
        min_object_size, min_hole_size, median_kernel = self.config_from_bboxarea(boxarea)
        if seg.rem_bbotblue():
            seg.mask = (reduce_holes(seg.mask, min_object_size=min_object_size, min_hole_size=min_hole_size).astype(np.uint8) * 255)
        seg.mask = cv2.medianBlur(seg.mask.astype(np.uint8), median_kernel)

        if seg.is_mask_empty() and box.cls["common_name"] != "colorchecker":
            return

        seg_props = self.get_segment_props(seg, rgb_crop, box)
        self.save_cutout_and_masks(imgdata, box, seg, seg_props, rgb_crop, semantic_mask_zeros, semantic_palette, instance_mask_zeros, instance_colors, instance_palette, cutout_ids, x1, y1, x2, y2)

    def scale_bbox(self, imgdata: ImageData, box: dict) -> Tuple[int, int, int, int]:
        """
        Scale bounding box coordinates based on image data.

        Args:
            imgdata (ImageData): Image dataclass object containing bounding boxes.
            box (dict): Dictionary representing a single bounding box.

        Returns:
            Tuple[int, int, int, int]: Scaled bounding box coordinates.
        """
        box.cls = self.fix_category(imgdata.batch_id, box.cls, box.bbox_id, imgdata.exif_meta.DateTime)
        scale = [imgdata.fullres_width, imgdata.fullres_height]
        _, x1, y1, x2, y2 = prep_bbox(box, scale)
        if any(val < 0 for val in (y1, y2, x1, x2)):
            coords = tuple(max(0, val) for val in (y1, y2, x1, x2))
            y1, y2, x1, x2 = coords[0], coords[1], coords[2], coords[3]
        return x1, y1, x2, y2

    def get_rgb_crop_and_data(self, rgb_array: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Get the cropped RGB array from the bounding box coordinates.

        Args:
            rgb_array (np.ndarray): Array representing the RGB image.
            x1 (int): Top-left x-coordinate.
            y1 (int): Top-left y-coordinate.
            x2 (int): Bottom-right x-coordinate.
            y2 (int): Bottom-right y-coordinate.

        Returns:
            np.ndarray: Cropped RGB array.
        """
        rgb_crop = rgb_array[y1:y2, x1:x2]
        return rgb_crop

    def is_valid_boxarea(self, boxarea: float, box: dict, seg: Segment, rgb_crop: np.ndarray, box_coords: Tuple[int, int, int, int]) -> bool:
        """
        Check if the bounding box area is valid based on thresholds.

        Args:
            boxarea (float): Area of the bounding box.
            box (dict): Dictionary representing a single bounding box.
            seg (Segment): Segment object for segmentation.
            rgb_crop (np.ndarray): Cropped RGB array.
            box_coords (Tuple[int, int, int, int]): Bounding box coordinates.

        Returns:
            bool: True if the box area is valid, False otherwise.
        """
        lower_bound, upper_bound = self.bbox_stats_calculator.outlier_thresholds.get(box.cls["common_name"], {})
        if boxarea < lower_bound:
            log.warning(f"Skipping cutout {box.bbox_id} due to small size({lower_bound}) for {box.cls['common_name']}")
            return False

        if boxarea > 15000000:
            log.debug(f"\n*******************\n Likely multiple plants in a single detection result. \n{box.bbox_id} - {boxarea} \nPrimary status: {box.is_primary} \n*************\n")

        if boxarea < 25000:
            seg.mask = seg.cotlydon()
        else:
            seg.mask = seg.general_seg(mode="cluster")

        if seg.is_mask_empty() and box.cls["common_name"] != "colorchecker":
            return False

        if boxarea > 200000:
            new_rgb_crop = apply_mask(rgb_crop, seg.mask, "black")
            seg = Segment(new_rgb_crop, species=box.cls, bbox=box_coords)
            seg.mask = seg.general_seg(mode="cluster")
        return True

    def get_segment_props(self, seg: Segment, rgb_crop: np.ndarray, box: dict) -> Dict:
        """
        Get properties of the segmented cutout.

        Args:
            seg (Segment): Segment object for segmentation.
            rgb_crop (np.ndarray): Cropped RGB array.
            box (dict): Dictionary representing a single bounding box.

        Returns:
            Dict: Dictionary containing segment properties.
        """
        seg_props = asdict(GenCutoutProps(rgb_crop, seg.mask).to_dataclass())
        seg_props["is_primary"] = box.is_primary
        seg_props["extends_border"] = seg.get_extends_borders(seg.mask)
        return seg_props

    def save_cutout_and_masks(self, imgdata: ImageData, box: dict, seg: Segment, seg_props: Dict, rgb_crop: np.ndarray, semantic_mask_zeros: np.ndarray, semantic_palette: List, instance_mask_zeros: np.ndarray, instance_colors: List, instance_palette: List, cutout_ids: List, x1: int, y1: int, x2: int, y2: int):
        """
        Save the cutout and associated masks.

        Args:
            imgdata (ImageData): Image dataclass object containing bounding boxes.
            box (dict): Dictionary representing a single bounding box.
            seg (Segment): Segment object for segmentation.
            seg_props (Dict): Dictionary containing segment properties.
            rgb_crop (np.ndarray): Cropped RGB array.
            semantic_mask_zeros (np.ndarray): Array representing the semantic mask.
            semantic_palette (List): List of semantic palette colors.
            instance_mask_zeros (np.ndarray): Array representing the instance mask.
            instance_colors (List): List of instance colors.
            instance_palette (List): List of instance palette colors.
            cutout_ids (List): List to store cutout IDs.
            x1 (int): Top-left x-coordinate.
            y1 (int): Top-left y-coordinate.
            x2 (int): Bottom-right x-coordinate.
            y2 (int): Bottom-right y-coordinate.
        """
        box.instance_rgb_id = generate_new_color(instance_colors, pastel_factor=0.7)
        box.cutout_exists = True
        cutout = Cutout(
            season=self.season,
            datetime=imgdata.exif_meta.DateTime,  # Use imgdata's datetime
            batch_id=self.batch_id,
            image_id=imgdata.image_id,  # Use imgdata's image_id
            cutout_id=box.bbox_id,
            cutout_num=int(box.bbox_id.split("_")[-1]),
            cutout_props=seg_props,
            cutout_height=rgb_crop.shape[0],
            cutout_width=rgb_crop.shape[1],
            category=box.cls
        )
        cutout_array = apply_mask(rgb_crop, seg.mask, "black")
        cutout_mask = np.zeros(seg.mask.shape[:2])
        cutout_mask[seg.mask != 0] = box.cls["class_id"]
        cutout.save_cutout(self.cutout_dir, cutout_array)
        cutout.save_config(self.cutout_dir)
        cutout.save_cropout(self.cutout_dir, rgb_crop)
        cutout.save_cutout_mask(self.cutout_dir, cutout_mask)
        cutout_ids.append(cutout.cutout_id)

        if box.cls["common_name"] != "colorchecker":
            semantic_mask_zeros[y1:y2, x1:x2] = cutout_mask
            semantic_palette.append(box.cls["class_id"])
            instance_mask_zeros[y1:y2, x1:x2][seg.mask == 255] = box.instance_rgb_id
            instance_colors.append(box.instance_rgb_id)
            instance_palette.append(box.instance_rgb_id)

    def cutout_pipeline(self, payload: dict) -> None:
        """
        Run the cutout processing pipeline on the provided payload.

        Args:
            payload (dict): Dictionary containing image data and other information.
        """
        imgdata = payload["imgdata"] if self.cfg.segment.multiprocess else payload
        log.info(f"Processing {imgdata.image_id}")

        rgb_array = imgdata.array
        bboxes = imgdata.bboxes

        instance_colors = [[0, 0, 0]]
        instance_palette = [[0, 0, 0]]
        semantic_palette = [[0]]

        cutout_ids, semantic_mask_zeros, instance_mask_zeros = self.process_bbox(
            imgdata, rgb_array, bboxes, instance_colors, instance_palette, semantic_palette
        )

        imgdata.cutout_ids = cutout_ids
        imgdata.save_config(self.metadata)
        imgdata.save_mask(self.semantic_mask_dir, semantic_mask_zeros)
        imgdata.save_mask(self.instance_mask_dir, instance_mask_zeros)

class VegetationSegmentationPipeline:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the vegetation segmentation pipeline.

        Args:
            cfg (DictConfig): Configuration object containing dataset and processing settings.
        """
        self.cfg = cfg
        self.dir_initializer = DirectoryInitializer(cfg)
        self.bbox_stats_calculator = BBoxStatsCalculator(cfg)
        self.cutout_processor = CutoutProcessor(cfg, self.bbox_stats_calculator, self.dir_initializer)

    def process(self):
        """
        Process the entire batch of images for vegetation segmentation.
        """
        batch_dir = Path(self.cfg.data.batchdir)
        cutoutdir = self.cfg.data.cutoutdir
        batch_id = self.cfg.general.batch_id
        csv_path = Path(cutoutdir, batch_id, f"{batch_id}.csv")

        metadir = Path(f"{self.cfg.data.batchdir}/metadata")
        if self.cfg.segment.multiprocess:
            return_list = create_dataclasses(metadir, self.cfg)
            payloads = [{"imgdata": img, "idx": idx} for idx, img in enumerate(return_list)]

            log.info(f"Multi-Processing image data for batch {batch_dir.name}.")
            procs = cpu_count() // 6
            with ProcessPoolExecutor(max_workers=procs) as executor:
                executor.map(self.cutout_processor.cutout_pipeline, payloads)
            log.info(f"Finished segmenting vegetation for batch {batch_dir.name}")
        else:
            log.info(f"Processing image data for batch {batch_dir.name}.")
            return_list = create_dataclasses(metadir, self.cfg)

            for imgdata in tqdm(return_list):
                self.cutout_processor.cutout_pipeline(imgdata)
            log.info(f"Finished segmenting vegetation for batch {batch_dir.name}")

        log.info(f"Condensing cutout results into a single csv file.")
        df = cutoutmeta2csv(cutoutdir, batch_id, csv_path, save_df=True)
        log.info(f"{len(df)} cutouts created.")

def main(cfg: DictConfig) -> None:
    """
    Main function to execute the vegetation segmentation pipeline.

    Args:
        cfg (DictConfig): Configuration object containing dataset and processing settings.
    """
    start = time.time()
    pipeline = VegetationSegmentationPipeline(cfg)
    pipeline.process()
    end = time.time()
    log.info(f"Segmentation completed in {end - start} seconds.")