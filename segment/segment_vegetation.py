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
from semif_utils.utils import calculate_bbox_area_cm2
from semif_utils.model import SegmentationModule, MaskPredictor
import concurrent.futures
from time import time
import torch
from typing import Any, Dict, Optional
from datetime import datetime
import os
import random
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utility functions for loading and processing images and JSON metadata
# -----------------------------------------------------------------------------
def load_json(file_path: Path) -> dict:
    with open(file_path, "r") as infile:
        return json.load(infile)


def load_rgb_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    # Ensure the image is contiguous in memory and convert to RGB
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_rgb_crop(rgb_array: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Return a cropped RGB image given bounding box coordinates."""
    return rgb_array[y1:y2, x1:x2]


def xywh2xyxy(bbox: List[int]) -> Tuple[int, int, int, int]:
    """Convert bounding box from [x, y, w, h] format to [x1, y1, x2, y2] format."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


# -----------------------------------------------------------------------------
# Functions for calculating bounding box statistics and outlier thresholds
# -----------------------------------------------------------------------------
def calculate_bbox_stats_per_species(metadata_jsons: Path, remap_species_info: Dict) -> Dict:
    """
    Calculate bounding box area statistics per species.
    """
    metadata_dicts = []
    bbox_areas_per_species = defaultdict(list)
    for metadata_json in metadata_jsons:
        metadata = load_json(metadata_json)
        annotations = metadata["annotations"]
        
        fullres_width = metadata["exif_meta"].get('ImageWidth')
        fullres_height = metadata["exif_meta"].get('ImageLength')
        image_height_m = metadata["camera_info"]["fov"].get('height')
        image_width_m  = metadata["camera_info"]["fov"].get('width')
        for annotation in annotations:
            bbox_xywh = annotation.get('bbox_xywh')
            cutout_width = bbox_xywh[2]
            cutout_height = bbox_xywh[3]
            
            annotation["bbox_area_cm2"] = calculate_bbox_area_cm2(
                image_height_m, image_width_m, 
                cutout_height, cutout_width, 
                fullres_width, fullres_height
                )
            global_boxarea = annotation["bbox_area_cm2"]
            category_class_id = annotation["category_class_id"]
            annotation_cat = remap_species_info[category_class_id]
            bbox_areas_per_species[annotation_cat["common_name"]].append(global_boxarea)
        
        metadata_dicts.append(metadata)

    bbox_stats = {}
    for species, areas in bbox_areas_per_species.items():
        valid_areas = [area for area in areas if area is not None]
        
        if len(valid_areas) == 0:
            log.warning(f"No valid areas found for {species}")
            bbox_stats[species] = {"areas": None, "mean": None, "25th_percentile": None}
        else:
            bbox_stats[species] = {
            "areas": valid_areas,
            "mean": np.mean(valid_areas),
            "25th_percentile": np.percentile(valid_areas, 25)
        }
    return bbox_stats, metadata_dicts

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
    

# -----------------------------------------------------------------------------
# Functions for segmentation mask creation and post-processing
# -----------------------------------------------------------------------------
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

def initialize_segmentation_predictor(cfg: DictConfig) -> MaskPredictor:
    """
    Initialize the segmentation module.
    """
    # model_checkpoint = Path(cfg.paths.inference.checkpoints, "best.ckpt")
    model_checkpoint = Path("/home/mkutuga/SemiF-Segmentation/projects/SEMIF_512/train/version_0/checkpoints/best.ckpt")
    
    model = SegmentationModule.load_from_checkpoint(
            arch_name="DeepLabV3Plus",
            encoder_name="resnet152",
            encoder_weights="imagenet",
            in_channels=3,
            out_classes=1,
            mode="binary",
            ignore_index=None,
            checkpoint_path=model_checkpoint,
        )
    
    # model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.to("cpu")
    mean = [0.4117603520173352, 0.39580040730759447, 0.30041100691031414] #cfg.inference.mean
    std = [0.18172717870741403, 0.17683208433640774, 0.14632597613213394] #cfg.inference.std
    
    predictor = MaskPredictor(
        model, 
        mean, 
        std, 
        use_normalization=True, 
        rescale_factor=1, 
        threshold=0.5
        )
    
    return predictor

def predict_mask_for_cutout(
    predictor: MaskPredictor,
    rgb_crop: np.ndarray,
    bbox_id: str,
    global_boxarea: float,
    lower_bound: float,
    upper_bound: float,
    category: Dict,
) -> np.ndarray:
    """
    Generate and post-process a segmentation mask for a given cutout.
    Returns:
        The post-processed mask (numpy array) or None if the cutout should be skipped.
    """
    mask = predictor.predict(rgb_crop)
    # Assign mask the correct class ID
    mask = mask.astype(np.uint8)
    
    # Post-processing: reduce holes and smooth the mask.
    min_object_size, min_hole_size, median_kernel = holeconfig_from_bboxarea(global_boxarea)

    mask = reduce_holes(mask, min_object_size, min_hole_size).astype(np.uint8)

    mask[mask == 1] = category["class_id"]
    mask = cv2.medianBlur(mask.astype(np.uint8), median_kernel)

    return mask
    
def generate_mask_for_cutout(
    seg: Segment,
    rgb_crop: np.ndarray,
    bbox_id: str,
    boxarea: float,
    lower_bound: float,
    upper_bound: float,
    category: Dict,
    bbox_coords: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Generate and post-process a segmentation mask for a given cutout.

    The logic is as follows:
      - If no box area is provided, use a general segmentation.
      - If the area is too small (below lower_bound), reject the cutout.
      - For very small objects (boxarea < 20), use an alternative segmentation (e.g. 'cotlydon').
      - For larger areas, use general segmentation.
      - If the mask is empty (and the object is not a "colorchecker"), reject the cutout.
      - For very large cutouts (boxarea > 1000), re-segment after applying the mask.
      - Apply hole reduction (if applicable) and median blur to clean up the mask.

    Returns:
        The post-processed mask (numpy array) or None if the cutout should be skipped.
    """
    # Initial segmentation based on bounding box area
    if boxarea is None:
        seg.mask = seg.general_seg(mode="cluster")
    else:
        if boxarea < lower_bound:
            log.warning(f"Skipping cutout {bbox_id}: box area {boxarea} below lower bound {lower_bound}")
            return None

        if boxarea < 20:
            seg.mask = seg.cotlydon()
        else:
            seg.mask = seg.general_seg(mode="cluster")

        if seg.is_mask_empty() and category["common_name"] != "colorchecker":
            log.warning(f"Skipping cutout {bbox_id}: generated mask is empty")
            return None

        if boxarea > 1000:
            # For very large regions, re-apply segmentation after masking the background.
            new_rgb_crop = apply_mask(rgb_crop, seg.mask, "black")
            seg = Segment(new_rgb_crop, species=category, bbox=bbox_coords)
            seg.mask = seg.general_seg(mode="cluster")

    # Post-processing: reduce holes and smooth the mask.
    min_object_size, min_hole_size, median_kernel = holeconfig_from_bboxarea(boxarea)
    
    if seg.rem_bbotblue():
        log.debug(f"Cutout {bbox_id}: removing blue background artifacts")
        seg.mask = reduce_holes(seg.mask, min_object_size, min_hole_size).astype(np.uint8) * 255

    log.debug(f"Cutout {bbox_id}: applying median blur with kernel size {median_kernel}")
    seg.mask = cv2.medianBlur(seg.mask.astype(np.uint8), median_kernel)

    if seg.is_mask_empty() and category["common_name"] != "colorchecker":
        log.warning(f"Skipping cutout {bbox_id}: mask empty after post-processing")
        return None

    return seg.mask

def get_segment_props(seg: Segment, rgb_crop: np.ndarray, box: dict, bbox_area_cm2: float) -> Dict:
    """
    Get properties of the segmented cutout.
    """
    seg_props = GenCutoutProps(rgb_crop, seg.mask).to_regprops_table()
    seg_props["is_primary"] = box["is_primary"]
    seg_props["extends_border"] = seg.get_extends_borders(seg.mask)
    seg_props["bbox_area_cm2"] = bbox_area_cm2
    seg_props["non_target_weed"] = None
    seg_props["non_target_weed_pred_conf"] = None
    return seg_props

def get_bbot_version(batch_id: str, date_ranges: Dict[str, Any]) -> Optional[str]:
    """
    Given a batch_id in the form "STATE_YYYY-MM-DD", this function extracts the state
    and date, then uses the date_ranges in the configuration (cfg) to determine and return
    the corresponding bbot_version.

    Args:
        batch_id (str): The batch identifier, e.g., "MD_2022-06-15".
        cfg (Dict[str, Any]): The configuration dictionary that contains a "date_ranges" key.

    Returns:
        Optional[str]: The bbot_version if a matching date range is found, otherwise None.
    """
    # Split the batch_id; expected format is "STATE_YYYY-MM-DD"
    parts = batch_id.split("_")
    if len(parts) < 2:
        raise ValueError("Batch ID must be in the format 'STATE_YYYY-MM-DD'")

    state = parts[0]
    date_str = parts[1]

    try:
        batch_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError("The date in batch_id must be in the format YYYY-MM-DD") from e

    # Retrieve the date ranges for the given state from the configuration
    state_ranges = date_ranges.get(state)
    if state_ranges is None:
        raise ValueError(f"State '{state}' not found in configuration date_ranges.")

    # Iterate over the date ranges for the state and find a matching date range
    for range_key, range_info in state_ranges.items():
        try:
            start_date = datetime.strptime(range_info["start"], "%Y-%m-%d").date()
            end_date = datetime.strptime(range_info["end"], "%Y-%m-%d").date()
        except (KeyError, ValueError) as e:
            # Skip any range that doesn't have properly formatted dates
            continue

        if start_date <= batch_date <= end_date:
            return range_info.get("bbot_version")

    # If no matching range was found, you can return None or raise an error
    return None
# -----------------------------------------------------------------------------
# Functions for saving outputs (cutouts, masks, metadata)
# -----------------------------------------------------------------------------
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
          date_ranges: Dict
          ):
        """
        Save the cutout and associated masks.
        """
        annotation["instance_rgb_id"] = generate_new_color(instance_colors, pastel_factor=0.7)
        annotation["cutout_exists"] = True
        
        cutout_dict = {
            "season": metadata["season"],
            "datetime": metadata["exif_meta"]["DateTime"],  # Use imgdata's datetime
            "bbot_version": get_bbot_version(metadata["batch_id"], date_ranges),
            "batch_id": metadata["batch_id"],
            "image_id": metadata["image_id"],  # Use imgdata's image_id
            "cutout_id": annotation["cutout_id"],
            "cutout_num": annotation["cutout_num"],
            "cutout_height": rgb_crop.shape[0],
            "cutout_width": rgb_crop.shape[1],
            "lens_model": metadata["exif_meta"]["LensModel"],
            "validated": False,
            "cutout_version": "1.0",
            "cutout_props": seg_props,
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
    """
    Map the local (cutout) masks back into full-size semantic and instance masks.
    """
    semantic_mask_zeros[y1:y2, x1:x2] = cutout_mask

    r, g, b = annotation["instance_rgb_id"]    
    instance_mask_zeros[y1:y2, x1:x2, 0] = np.where((instance_mask_zeros[y1:y2, x1:x2, 0] == 0) & (seg.mask != 0), r, instance_mask_zeros[y1:y2, x1:x2, 0])
    instance_mask_zeros[y1:y2, x1:x2, 1] = np.where((instance_mask_zeros[y1:y2, x1:x2, 1] == 0) & (seg.mask != 0), g, instance_mask_zeros[y1:y2, x1:x2, 1])
    instance_mask_zeros[y1:y2, x1:x2, 2] = np.where((instance_mask_zeros[y1:y2, x1:x2, 2] == 0) & (seg.mask != 0), b, instance_mask_zeros[y1:y2, x1:x2, 2])

    semantic_palette.append(annotation["category_class_id"])
    instance_colors.append(annotation["instance_rgb_id"])
    instance_palette.append(annotation["instance_rgb_id"])

# -----------------------------------------------------------------------------
# Main processing function for each metadata file
# -----------------------------------------------------------------------------
def process_metadata_file(args: Tuple[Path, Path, str, Dict, Dict, Path, Path, Path]) -> None:
    """
    Process a single metadata JSON file.
    """
    # time this function
    start = time()
    global predictor  # Ensure we use the global predictor loaded by the initializer
    (metadata, image_dir, season, remap_species_info, date_ranges,
     outlier_thresholds, cutout_dir, semantic_mask_dir, instance_mask_dir) = args

    metadata_path = Path("data", "semifield-developed-images", metadata["batch_id"], "metadata", metadata["image_id"] + ".json")
    log.debug(f'Processing {metadata["image_id"]}')
    
    img_path = image_dir / f"{metadata['image_id']}.jpg"
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
        global_boxarea = annotation.pop("bbox_area_cm2")
        
        # Create a rule that if the boundinbg box is the size of the image, skip it
        if annotation["bbox_xywh"][2] == rgb_array.shape[1] and annotation["bbox_xywh"][3] == rgb_array.shape[0]:
            log.debug(f"Skipping annotation {annotation['cutout_id']} with full image bounding box")
            continue
        
        if "bbox_xywh" in annotation:    
            annotation["cutout_id"] = f"{metadata['image_id']}_{idx}"
            annotation["cutout_num"] = idx
            x1, y1, x2, y2 = xywh2xyxy(annotation["bbox_xywh"])
            rgb_crop = get_rgb_crop(rgb_array, x1, y1, x2, y2)

            category_class_id = annotation["category_class_id"]
            annotation_cat = remap_species_info[category_class_id]
            
            # Create a segmentation object for the cutout.
            seg = Segment(rgb_crop, species=annotation_cat, bbox=(x1, y1, x2, y2))
            lower_bound, upper_bound = outlier_thresholds.get(annotation_cat["common_name"], (0, np.inf))

            # Generate and post-process the mask.
            # seg.mask = generate_mask_for_cutout(seg, rgb_crop, annotation["cutout_id"], global_boxarea, lower_bound, upper_bound, annotation_cat, (x1, y1, x2, y2))
            
            # Use the learning-based predictor to generate the mask.
            seg.mask = predict_mask_for_cutout(predictor, rgb_crop, annotation["cutout_id"], global_boxarea, lower_bound, upper_bound, annotation_cat)
            if seg.is_mask_empty():
                continue  # Skip this cutout if the mask is invalid
            
            # Extract segmentation properties and save outputs.
            seg_props = get_segment_props(seg, rgb_crop, annotation, global_boxarea)
            cutout_mask = save_cutout_and_masks(
                cutout_dir,
                metadata,
                annotation,
                remap_species_info,
                seg,
                seg_props,
                rgb_crop,
                instance_colors,
                cutout_ids,
                date_ranges
            )

            # Update full-sized masks if not a colorchecker.
            if annotation_cat["common_name"] != "colorchecker":
                map_fullsized_masks(
                    semantic_mask_zeros,
                    instance_mask_zeros,
                    cutout_mask,
                    annotation,
                    semantic_palette,
                    instance_colors,
                    instance_palette,
                    x1,
                    y1,
                    x2,
                    y2,
                    seg,
                    idx,
                )

    save_metadata(metadata_path, metadata)
    save_mask(semantic_mask_dir / f"{metadata['image_id']}.png", semantic_mask_zeros)
    save_instance_mask(instance_mask_dir / f"{metadata['image_id']}.png", instance_mask_zeros)
    end = time()
    # log duration in seconds and minutes using 3 decimal places
    log.info(f"Processed {metadata['image_id']} in {end - start:.3f} seconds ({(end - start) / 60:.3f} minutes)")

# Global variable to store the predictor instance.
predictor = None
# -----------------------------------------------------------------------------
# Execution modes: sequential or parallel processing of metadata files.
# -----------------------------------------------------------------------------

def init_worker(cfg: DictConfig) -> None:
    # Set thread limits for underlying libraries in each worker.
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    global predictor
    predictor = initialize_segmentation_predictor(cfg)
    log.info(f"[Worker {os.getpid()}] Predictor initialized with limited threads.")

def run_sequential(args_list: List[Tuple], cfg: DictConfig):
    """
    Run the processing sequentially (using a for loop) while ensuring that
    the learning-based predictor is loaded only once.
    """
    global predictor
    # Initialize the predictor if it has not been loaded yet.
    if predictor is None:
        predictor = initialize_segmentation_predictor(cfg)
        log.info("Predictor initialized in sequential mode.")
    
    for args in tqdm(args_list):
        log.info(f"Processing {args[0]['image_id']} sequentially")
        process_metadata_file(args)

def run_parallel(args_list, cfg: DictConfig):
    max_workers = 8  # Limit the number of processes to avoid using all CPUs.
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(cfg,)) as executor:
        # results = list(tqdm(executor.map(process_metadata_file, args_list), total=len(args_list)))
        results = list(executor.map(process_metadata_file, args_list))
    return results

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main(cfg: DictConfig) -> None:
    """
    Main function to execute the vegetation segmentation pipeline.
    """

    season = cfg.general.season
    species_info = load_json(cfg.data.species)
    remap_species_info = {details['class_id']: details for _, details in species_info["species"].items()}
    [v.pop(key) for v in remap_species_info.values() for key in ["collection_timing", "collection_location"]]
    batch_id = cfg.general.batch_id

    image_dir = Path(cfg.batchdata.images)
    metadata_dir = Path(cfg.batchdata.metadata)
    semantic_mask_dir = Path(cfg.batchdata.meta_masks, "semantic_masks")
    instance_mask_dir = Path(cfg.batchdata.meta_masks, "instance_masks")
    cutout_dir = Path(cfg.batchdata.cutouts)
    date_ranges = cfg.date_ranges

    for directory in [cutout_dir, semantic_mask_dir, instance_mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(metadata_dir.glob("*.json"))[29:]
    bbox_stats, metadata_dicts = calculate_bbox_stats_per_species(metadata_paths, remap_species_info)
    outlier_thresholds = calculate_outlier_thresholds(bbox_stats)
    
    # Prepare arguments for parallel/sequential execution
    args_list = [
        (metadata, image_dir, season, remap_species_info, date_ranges, outlier_thresholds, 
         cutout_dir, semantic_mask_dir, instance_mask_dir)
        for metadata in metadata_dicts
    ]

    # Run sequentially or in parallel based on the `parallel` argument
    parallel = True

    if parallel:
        run_parallel(args_list, cfg)
    else:
        run_sequential(args_list, cfg)

    cutout_csv_path = cutout_dir / f"{batch_id}.csv"
    cutoutmeta2csv(cutout_dir.parent, batch_id, cutout_csv_path, save_df=True)

