import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from omegaconf import DictConfig
from scipy.stats import zscore
from tqdm import tqdm
from pprint import  pprint
from semif_utils.segment_species import Segment
from semif_utils.segment_utils import GenCutoutProps, generate_new_color
from semif_utils.utils import apply_mask, cutoutmeta2csv, reduce_holes
import concurrent.futures
import os
import random
log = logging.getLogger(__name__)


def calculate_bbox_stats_per_species(metadata_jsons: Path, remap_species_info: Dict) -> Dict:
    """
    Calculate bounding box area statistics per species.
    """
    bbox_areas_per_species = defaultdict(list)
    for metadata_json in metadata_jsons:
        log.debug(f"Processing {metadata_json}")
        metadata = load_json(metadata_json)
        annotations = metadata["annotations"]

        for annotation in annotations:
            global_boxarea = annotation["bbox_area_cm2"]
            # if global_boxarea is None:
            #     continue
            category_class_id = annotation["category_class_id"]
            annotation_cat = remap_species_info[category_class_id]
            
            bbox_areas_per_species[annotation_cat["common_name"]].append(global_boxarea)

    bbox_stats = {}
    for species, areas in bbox_areas_per_species.items():
        valid_areas = [area for area in areas if area is not None]
        
        if len(valid_areas) == 0:
            log.warning(f"No valid areas found for {species}")
            bbox_stats[species] = {
                "areas": None,
                "mean": None,
                "25th_percentile": None
            }
        else:
            bbox_stats[species] = {
            "areas": valid_areas,
            "mean": np.mean(valid_areas),
            "25th_percentile": np.percentile(valid_areas, 25)
        }
    return bbox_stats

def calculate_outlier_thresholds(bbox_stats) -> Dict:
    """
    Calculate outlier thresholds for bounding box areas.
    """
    outlier_thresholds = {}
    for species, stats in bbox_stats.items():
        if stats["areas"] is None:
            lower_bound = None
            upper_bound = None
        else:
            areas = np.array(stats["areas"])
            z_scores = zscore(areas)
            threshold = 2.0  # Z-score threshold for defining outliers
            lower_mask = z_scores < -threshold
            upper_mask = z_scores > threshold

            lower_bound = float(np.min(areas[~lower_mask]) if (~lower_mask).any() else 0 )
            upper_bound = float(np.max(areas[~upper_mask]) if (~upper_mask).any() else np.inf)

        outlier_thresholds[species] = (lower_bound, upper_bound)
    return outlier_thresholds
    
def load_json(species_path) -> dict:
        with open(species_path, "r") as outfile:
            return json.load(outfile)
        
def load_rgb_image(image_path: Path) -> np.ndarray:

    img_array = cv2.imread(str(image_path))
    img_array = np.ascontiguousarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    return img_array

def get_rgb_crop_and_data(rgb_array: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Get the cropped RGB array from the bounding box coordinates.
        """
        rgb_crop = rgb_array[y1:y2, x1:x2]
        return rgb_crop
        
def xywh2xyxy(bbox: List[int]) -> List[int]:
    """
    Convert bounding box from xywh format to xyxy format.
    """
    x, y, w, h = bbox
    return x, y, x + w, y + h

def is_valid_boxarea(bbox_id: str, rgb_crop: np.array, category: Dict, box_coords: Tuple, boxarea: float, lower_bound: float, upper_bound: float, seg: Segment) -> bool:
    """
    Check if the bounding box area is valid based on thresholds.
    """
    if boxarea is None:
        seg.mask = seg.general_seg(mode="cluster")
        if seg.is_mask_empty() and category["common_name"] != "colorchecker":
            log.warning(f"Skipping cutout {bbox_id} due to empty mask.")
            return False
        return True
    
    if boxarea < lower_bound:
        log.warning(f"Skipping cutout {bbox_id} due to small size({lower_bound}) for {category['common_name']}")
        return False

    if boxarea > 10000:  # Converted from 15000000 pixels to cm^2
        log.debug(f"\n*******************\n Likely multiple plants in a single detection result. \n{bbox_id} - {boxarea}\n*************\n")

    if boxarea < 20:  # Converted from 25000 pixels to cm^2, about the size of a small sticky note
        seg.mask = seg.cotlydon()
    else:
        seg.mask = seg.general_seg(mode="cluster")

    if seg.is_mask_empty() and category["common_name"] != "colorchecker":
        return False

    if boxarea > 1000:  # Converted from 200000 pixels to cm^2, about the size of A4 paper
        new_rgb_crop = apply_mask(rgb_crop, seg.mask, "black")
        seg = Segment(new_rgb_crop, species=category, bbox=box_coords)
        seg.mask = seg.general_seg(mode="cluster")
    return True

def holeconfig_from_bboxarea(boxarea: float) -> Tuple[int, int, int]:
        """
        Return configuration parameters based on bounding box area.
        """
        if boxarea is None:
            return 500, 500, 3   
        elif boxarea < 1: # About the size of a button
            return 100, 100, 1
        elif boxarea < 10: # About the size of a postage stamp
            return 500, 500, 3
        elif boxarea < 100: # About the size of a notecard
            return 1000, 1000, 7
        elif boxarea < 1000: # About the size of A4 paper
            return 5000, 5000, 9
        else:
            return 10000, 10000, 11
        
def get_segment_props(seg: Segment, rgb_crop: np.ndarray, box: dict) -> Dict:
        """
        Get properties of the segmented cutout.

        Args:
            seg (Segment): Segment object for segmentation.
            rgb_crop (np.ndarray): Cropped RGB array.
            box (dict): Dictionary representing a single bounding box.

        Returns:
            Dict: Dictionary containing segment properties.
        """
        seg_props = GenCutoutProps(rgb_crop, seg.mask).to_regprops_table()

        seg_props["is_primary"] = box["is_primary"]
        seg_props["extends_border"] = seg.get_extends_borders(seg.mask)
        seg_props["bbox_area_cm2"] = box.pop("bbox_area_cm2")
        return seg_props

def save_cutout(cutout_path, cutout_array):
    cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout_array, cv2.COLOR_RGB2BGRA))

def save_metadata(save_cutout_path: Path, metadata: Dict) -> bool:
    with open(save_cutout_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)

def save_cropout(cropout_path, img_array):
    cv2.imwrite(
        str(cropout_path),
        cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 100],
    )

def save_mask(mask_path, mask):
    cv2.imwrite(str(mask_path), mask)
    
def save_instance_mask(instance_mask_path, instance_mask):
    cv2.imwrite(str(instance_mask_path), cv2.cvtColor(instance_mask, cv2.COLOR_RGB2BGR))

    
def save_cutout_and_masks(
          cutout_dir: Path, 
          metadata: Dict, 
          annotation: Dict, 
          remap_species_info: Dict, 
          seg: Segment, 
          seg_props: Dict, 
          rgb_crop: np.ndarray, 
          instance_colors: List, 
          cutout_ids: List, 
          ):
        """
        Save the cutout and associated masks.
        """
        annotation["instance_rgb_id"] = generate_new_color(instance_colors, pastel_factor=0.7)
        annotation["cutout_exists"] = True
        
        cutout_dict = {
            "season": metadata["season"],
            "datetime": metadata["exif_meta"]["DateTime"],  # Use imgdata's datetime
            "batch_id": metadata["batch_id"],
            "image_id": metadata["image_id"],  # Use imgdata's image_id
            "cutout_id": annotation["cutout_id"],
            "cutout_num": annotation["cutout_num"],
            "cutout_props": seg_props,
            "cutout_height": rgb_crop.shape[0],
            "cutout_width": rgb_crop.shape[1],
            "category": remap_species_info[annotation["category_class_id"]],
        }
        
        
        cutout_array = apply_mask(rgb_crop, seg.mask, "black")
        cutout_mask = np.zeros(seg.mask.shape[:2])
        cutout_mask[seg.mask != 0] = int(annotation["category_class_id"])
       
        cutout_path = cutout_dir / f"{annotation['cutout_id']}.png"
        save_cutout(cutout_path, cutout_array)
        
        cutout_metadata_path = cutout_dir / f"{annotation['cutout_id']}.json"
        save_metadata(cutout_metadata_path, cutout_dict)
        
        cropout_path = cutout_dir / f"{annotation['cutout_id']}.jpg"
        save_cropout(cropout_path, rgb_crop)

        mask_path = cutout_dir / f"{annotation['cutout_id']}_mask.png"
        save_mask(mask_path, cutout_mask)
        
        cutout_ids.append(annotation["cutout_id"])

        return cutout_mask

def map_fullsized_masks(
          semantic_mask_zeros: np.ndarray,
          instance_mask_zeros: np.ndarray,
          cutout_mask: np.ndarray,
          annotation: Dict,
          semantic_palette: List,
          instance_colors: List,
          instance_palette: List,
          x1: int, y1: int, x2: int, y2: int,
          seg: Segment, 
          idx: int
          ):
    semantic_mask_zeros[y1:y2, x1:x2] = cutout_mask

    r, g, b = annotation["instance_rgb_id"]    
    instance_mask_zeros[y1:y2, x1:x2, 0] = np.where((instance_mask_zeros[y1:y2, x1:x2, 0] == 0) & (seg.mask != 0), r, instance_mask_zeros[y1:y2, x1:x2, 0])
    instance_mask_zeros[y1:y2, x1:x2, 1] = np.where((instance_mask_zeros[y1:y2, x1:x2, 1] == 0) & (seg.mask != 0), g, instance_mask_zeros[y1:y2, x1:x2, 1])
    instance_mask_zeros[y1:y2, x1:x2, 2] = np.where((instance_mask_zeros[y1:y2, x1:x2, 2] == 0) & (seg.mask != 0), b, instance_mask_zeros[y1:y2, x1:x2, 2])

    semantic_palette.append(annotation["category_class_id"])
    instance_colors.append(annotation["instance_rgb_id"])
    instance_palette.append(annotation["instance_rgb_id"])

    
def process_metadata_file(args: Tuple[Path, Path, str, Dict, Dict, Path, Path, Path]) -> None:
    """
    Process a single metadata JSON file.
    """
    (metadata_path, image_dir, season, remap_species_info, 
     outlier_thresholds, cutout_dir, semantic_mask_dir, instance_mask_dir) = args

    log.debug(f"Processing {metadata_path}")
    metadata = load_json(metadata_path)
    img_path = image_dir / f"{metadata['image_id']}.jpg"
    log.debug(f"Loading image {img_path}")
    rgb_array = load_rgb_image(img_path)

    metadata["season"] = season
    annotations = metadata["annotations"]

    instance_colors = [[0, 0, 0]]
    instance_palette = [[0, 0, 0]]
    semantic_palette = [[0]]

    cutout_ids = []
    semantic_mask_zeros = np.zeros(rgb_array.shape[:2], dtype=np.float32)
    instance_mask_zeros = np.zeros(rgb_array.shape, dtype=np.uint8)

    for idx, annotation in enumerate(annotations):
        if "bbox_xywh" in annotation:
            annotation["cutout_id"] = f"{metadata['image_id']}_{idx}"
            annotation["cutout_num"] = idx
            global_boxarea = annotation["bbox_area_cm2"]

            log.debug(f"Processing cutout {annotation['cutout_id']}")
            x1, y1, x2, y2 = xywh2xyxy(annotation["bbox_xywh"])
            log.debug(f"Bounding box coordinates: {x1}, {y1}, {x2}, {y2}")
            log.debug(f"Bounding box area: {global_boxarea}")
            log.debug(f"Category class ID: {annotation['category_class_id']}")
            rgb_crop = get_rgb_crop_and_data(rgb_array, x1, y1, x2, y2)


            category_class_id = annotation["category_class_id"]
            log.debug(f"Category class ID: {category_class_id}")
            annotation_cat = remap_species_info[category_class_id]
            log.debug(f"Category: {annotation_cat['common_name']}")
            seg = Segment(rgb_crop, species=annotation_cat, bbox=(x1, y1, x2, y2))

            lower_bound, upper_bound = outlier_thresholds.get(annotation_cat["common_name"], (0, np.inf))

            if not is_valid_boxarea(annotation["cutout_id"], rgb_crop, annotation_cat, (x1, y1, x2, y2), 
                                    global_boxarea, lower_bound, upper_bound, seg):
                continue

            min_object_size, min_hole_size, median_kernel = holeconfig_from_bboxarea(global_boxarea)

            if seg.rem_bbotblue():
                log.debug("Removing bbot blue")
                seg.mask = reduce_holes(seg.mask, min_object_size, min_hole_size).astype(np.uint8) * 255

            log.debug("Applying median blur")
            seg.mask = cv2.medianBlur(seg.mask.astype(np.uint8), median_kernel)

            log.debug("Getting segment properties")
            seg_props = get_segment_props(seg, rgb_crop, annotation)
            log.debug("Saving cutout and masks")
            cutout_mask = save_cutout_and_masks(
                cutout_dir, metadata, annotation, remap_species_info, seg, seg_props, rgb_crop, 
                instance_colors, cutout_ids)

            if annotation_cat["common_name"] != "colorchecker":
                map_fullsized_masks(
                    semantic_mask_zeros, instance_mask_zeros, cutout_mask, annotation, 
                    semantic_palette, instance_colors, instance_palette, x1, y1, x2, y2, seg, idx)

    metadata["cutout_ids"] = cutout_ids

    save_metadata(metadata_path, metadata)
    save_mask(semantic_mask_dir / f"{metadata['image_id']}.png", semantic_mask_zeros)
    save_instance_mask(instance_mask_dir / f"{metadata['image_id']}.png", instance_mask_zeros)


def run_sequential(args_list: List[Tuple]):
    """
    Run the processing sequentially (for loop).
    """
    for args in args_list:
        log.info(f"Processing {args[0]}")
        process_metadata_file(args)


def run_parallel(args_list: List[Tuple]):
    """
    Run the processing in parallel using concurrent.futures.
    """
    max_workers = int(len(os.sched_getaffinity(0)) / 5)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_metadata_file, args_list)

def main(cfg: DictConfig) -> None:
    """
    Main function to execute the vegetation segmentation pipeline.
    """

    season = cfg.general.season
    species_info = load_json(cfg.data.species)
    remap_species_info = {details['class_id']: details for _, details in species_info["species"].items()}
    batch_id = cfg.general.batch_id

    image_dir = Path(cfg.batchdata.images)
    metadata_dir = Path(cfg.batchdata.metadata)
    semantic_mask_dir = Path(cfg.batchdata.meta_masks, "semantic_masks")
    instance_mask_dir = Path(cfg.batchdata.meta_masks, "instance_masks")
    cutout_dir = Path(cfg.batchdata.cutouts)

    for directory in [cutout_dir, semantic_mask_dir, instance_mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(metadata_dir.glob("*.json"))
    bbox_stats = calculate_bbox_stats_per_species(metadata_paths, remap_species_info)
    outlier_thresholds = calculate_outlier_thresholds(bbox_stats)

    random.shuffle(metadata_paths)
    
    # Prepare arguments for parallel/sequential execution
    args_list = [
        (metadata_path, image_dir, season, remap_species_info, outlier_thresholds, 
         cutout_dir, semantic_mask_dir, instance_mask_dir)
        for metadata_path in metadata_paths
    ]

    # Run sequentially or in parallel based on the `parallel` argument
    parallel = True

    if parallel:
        run_parallel(args_list)
    else:
        run_sequential(args_list)

    cutout_csv_path = cutout_dir / f"{batch_id}.csv"
    cutoutmeta2csv(cutout_dir.parent, batch_id, cutout_csv_path, save_df=True)

