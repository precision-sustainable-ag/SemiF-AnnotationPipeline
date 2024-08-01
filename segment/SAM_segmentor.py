import os
import json
import math
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes, label

from segment_anything_hq import sam_model_registry, SamPredictor

log = logging.getLogger(__name__)

device = "cuda"
torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner

class DirectoryInitializer:
    def __init__(self, cfg: DictConfig):
        self.batch_id = cfg.general.batch_id
        self.image_dir = Path(cfg.batchdata.images)
        self.metadata_dir = Path(cfg.batchdata.metadata)
        self.cutout_dir = Path(cfg.data.cutoutdir)
        self.meta_mask_dir = Path(cfg.batchdata.meta_masks)

        self._initialize_directories()

    def _initialize_directories(self):
        self.semantic_mask_dir = self.meta_mask_dir / "semantic_masks"
        self.instance_mask_dir = self.meta_mask_dir / "instance_masks"
        self.vis_masks = self.meta_mask_dir / "vis_masks"
        self.cutout_batch_dir = self.cutout_dir / self.batch_id

        for directory in [self.cutout_batch_dir, self.semantic_mask_dir, self.instance_mask_dir, self.vis_masks]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_images(self) -> List[Path]:
        return sorted(self.image_dir.glob("*.jpg"))

class SingleImageProcessor:
    def __init__(self, cfg: DictConfig, directory_initializer: DirectoryInitializer):
        log.info("Initializing SingleImageProcessor")
        self.instance_label_dir = directory_initializer.instance_mask_dir
        self.class_label_dir = directory_initializer.semantic_mask_dir
        self.visualization_label_dir = directory_initializer.vis_masks

def save_compressed_image(image: np.ndarray, path: str, quality: int = 98):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    is_success, encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if is_success:
        with open(path, 'wb') as f:
            encoded_image.tofile(f)


def process_single_image(input_paths: tuple, instance_label_dir: Path, class_label_dir: Path, visualization_label_dir: Path, cfg: DictConfig):
    log.info("Loading SAM model")
    sam = sam_model_registry[cfg.segment.sam.model_type](checkpoint=cfg.segment.sam.sam_checkpoint)
    sam.to(device=device)
    mask_predictor = SamPredictor(sam)

    image_path, json_path = input_paths
    log.info(f"Processing image: {image_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = pd.DataFrame(data['annotations'])
    if annotations.empty:
        log.warning(f"No annotations found for image {image_path}")
        return
    
    log.info(f"Processing {annotations.shape[0]} annotations")
    annotations[['classID', 'minX', 'minY', 'width', 'height']] = annotations.apply(
        lambda x: [x['category_class_id'], *x['bbox_xywh']], axis=1, result_type='expand')
    
    annotations['maxX'] = annotations['minX'] + annotations['width']
    annotations['maxY'] = annotations['minY'] + annotations['height']

    process_annotations(annotations)
    
    image = read_image(image_path)
    
    masked_image, class_masked_image, instance_masked_image = create_masks(image, annotations, mask_predictor)
    
    mask_name = Path(image_path).stem + '.png'
    log.info(f"Saving masks ({Path(image_path).stem}) to {visualization_label_dir.parent}")
    save_compressed_image(masked_image, visualization_label_dir / mask_name)
    cv2.imwrite(str(instance_label_dir / mask_name), instance_masked_image.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cv2.imwrite(str(class_label_dir / mask_name), class_masked_image.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])

def process_annotations(annotations: pd.DataFrame):
    annotations['centerX'] = (annotations['minX'] + annotations['maxX']) / 2
    annotations['centerY'] = (annotations['minY'] + annotations['maxY']) / 2
    
def read_image(image_path: Path) -> np.ndarray:
    log.info(f"Reading image and converting to RGB: {image_path}")
    return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

def create_masks(image: np.ndarray, annotations: pd.DataFrame, mask_predictor: SamPredictor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    im_size_X, im_size_Y = image.shape[1], image.shape[0]
    im_pad_size = 1500
    
    image_expanded = cv2.copyMakeBorder(image, im_pad_size, im_pad_size, im_pad_size, im_pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    masked_image = np.copy(image_expanded)
    class_masked_image = np.ones(masked_image.shape[0:2]) * 255
    instance_masked_image = np.ones(masked_image.shape[0:2]) * 65535

    masked_image_rgba = np.zeros((masked_image.shape[0], masked_image.shape[1], 4), dtype=np.uint8)
    masked_image_rgba[..., :3] = masked_image

    update_annotations_with_absolute_coords(annotations)
    
    annotations = annotations.sort_values(by=['plant_bboxes_area'], ascending=False)

    for index, annotation in annotations.iterrows():
        process_annotation(annotation, image_expanded, masked_image_rgba, class_masked_image, instance_masked_image, im_pad_size, index, im_size_X, im_size_Y, mask_predictor)

    return masked_image_rgba[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size, :], class_masked_image[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size], instance_masked_image[im_pad_size:-im_pad_size, im_pad_size:-im_pad_size]

def update_annotations_with_absolute_coords(annotations: pd.DataFrame):
    annotations['plant_bboxes_area'] = annotations['width'] * annotations['height']

def process_annotation(annotation: pd.Series, image_expanded: np.ndarray, masked_image_rgba: np.ndarray, 
                        class_masked_image: np.ndarray, instance_masked_image: np.ndarray, 
                        im_pad_size: int, index: int, im_size_X: int, im_size_Y: int, mask_predictor: SamPredictor):
    plant_bbox = np.array([int(annotation['minX']), int(annotation['minY']), 
                           int(annotation['maxX']), int(annotation['maxY'])])

    sam_crop_size_x, sam_crop_size_y = determine_crop_size(annotation)
    
    cropped_image = crop_image(image_expanded, annotation, im_pad_size, sam_crop_size_x, sam_crop_size_y)
    
    mask_predictor.set_image(cropped_image)
    log.info(f"Cropped image size for SAM predictor: {cropped_image.shape}")
    
    _, cropped_bbox = get_bounding_boxes(annotation, plant_bbox, im_pad_size, sam_crop_size_x, sam_crop_size_y)
    input_boxes = torch.tensor(cropped_bbox, device=mask_predictor.device)
    
    transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, cropped_image.shape[:2])

    masks, _, _ = mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, 
                                               multimask_output=True, hq_token_only=False)
    
    apply_masks(masks, masked_image_rgba, class_masked_image, instance_masked_image, annotation, im_pad_size, sam_crop_size_x, sam_crop_size_y, index)

def determine_crop_size(annotation: pd.Series) -> tuple[int, int]:
    sam_crop_size_x = 1000
    sam_crop_size_y = 1000
    if annotation['width'] > 700:
        sam_crop_size_x = math.ceil(annotation['width'] * 1.43 / 2.) * 2
    if annotation['height'] > 700:
        sam_crop_size_y = math.ceil(annotation['height'] * 1.43 / 2.) * 2
    return sam_crop_size_x, sam_crop_size_y

def crop_image(image_expanded: np.ndarray, annotation: pd.Series, im_pad_size: int, 
                sam_crop_size_x: int, sam_crop_size_y: int) -> np.ndarray:
    return np.copy(image_expanded[int(annotation['centerY'] + im_pad_size - sam_crop_size_y / 2):int(annotation['centerY'] + im_pad_size + sam_crop_size_y / 2), 
                                   int(annotation['centerX'] + im_pad_size - sam_crop_size_x / 2):int(annotation['centerX'] + im_pad_size + sam_crop_size_x / 2), :])

def get_bounding_boxes(annotation: pd.Series, plant_bbox: np.ndarray, im_pad_size: int, 
                        sam_crop_size_x: int, sam_crop_size_y: int) -> tuple[np.ndarray, np.ndarray]:
    padded_bbox = plant_bbox + [im_pad_size, im_pad_size, im_pad_size, im_pad_size]
    cropped_bbox = padded_bbox - [annotation['centerX'] + im_pad_size - sam_crop_size_x / 2, 
                                  annotation['centerY'] + im_pad_size - sam_crop_size_y / 2, 
                                  annotation['centerX'] + im_pad_size - sam_crop_size_x / 2, 
                                  annotation['centerY'] + im_pad_size - sam_crop_size_y / 2]
    return padded_bbox, cropped_bbox

def make_exg(rgb_image: np.ndarray, normalize: bool = False, thresh: int = 0) -> np.ndarray:
    rgb_image = rgb_image.astype(float)
    r, g, b = cv2.split(rgb_image)

    if normalize:
        total = r + g + b
        total[total == 0] = 1
        exg = 2 * (g / total) - (r / total) - (b / total)
    else:
        exg = 2 * g - r - b

    if thresh is not None and not normalize:
        exg = np.where(exg < thresh, 0, exg)

    return exg.astype("uint8")

def clean_mask(mask: np.ndarray, cropped_image_area: np.ndarray) -> np.ndarray:
    kernel_size = {
        6: (5, 5),  
        28: (3, 3)  
    }
    kernel = np.ones((5, 5), np.uint8)

    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    cleaned_mask = remove_small_holes(cleaned_mask.astype(bool), area_threshold=100).astype(np.uint8)

    exg_image = make_exg(cropped_image_area)

    if exg_image.size == 0:
        return cleaned_mask

    labeled_mask, num = label(cleaned_mask, return_num=True)
    props = regionprops(labeled_mask)
    for prop in props:
        if prop.area > 500:  
            minr, minc, maxr, maxc = prop.bbox
            component_exg = exg_image[minr:maxr, minc:maxc]
            component_mask = (labeled_mask[minr:maxr, minc:maxc] == prop.label)

            if component_exg.size == 0 or component_mask.size == 0:
                continue

            mean_exg = np.mean(component_exg[component_mask])
            if mean_exg < 0:
                cleaned_mask[labeled_mask == prop.label] = 0

    return cleaned_mask

def apply_masks(masks: torch.Tensor, masked_image_rgba: np.ndarray, class_masked_image: np.ndarray, 
                 instance_masked_image: np.ndarray, annotation: pd.Series, im_pad_size: int, 
                 sam_crop_size_x: int, sam_crop_size_y: int, index: int):
    bb_color = tuple(np.random.random(size=3) * 255)
    for mask in masks:
        full_mask = np.zeros(masked_image_rgba.shape[0:2])
        
        crop_start_y = int(annotation['centerY'] + im_pad_size - sam_crop_size_y / 2)
        crop_end_y = int(annotation['centerY'] + im_pad_size + sam_crop_size_y / 2)
        crop_start_x = int(annotation['centerX'] + im_pad_size - sam_crop_size_x / 2)
        crop_end_x = int(annotation['centerX'] + im_pad_size + sam_crop_size_x / 2)
        
        full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = mask.cpu()[0, :, :]

        cropped_image_area = masked_image_rgba[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :3]
        cropped_mask = full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

        cleaned_cropped_mask = cropped_mask
        
        full_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = cleaned_cropped_mask

        alpha = 0.5
        for c in range(3):
            masked_image_rgba[full_mask == 1, c] = (1 - alpha) * masked_image_rgba[full_mask == 1, c] + alpha * bb_color[c]
        masked_image_rgba[full_mask == 1, 3] = int(alpha * 255)

        class_masked_image[full_mask == 1] = annotation['classID']
        instance_masked_image[full_mask == 1] = index

def process_sequentially(directory_initializer: DirectoryInitializer, processor: SingleImageProcessor, cfg: DictConfig):
    imgs = directory_initializer.get_images()

    for image_path in imgs:
        log.info(f"Processing image: {image_path}")
        json_path = str(directory_initializer.metadata_dir / f"{image_path.stem}.json")
        input_paths = (image_path, json_path)
        process_single_image(input_paths, processor.instance_label_dir, processor.class_label_dir, processor.visualization_label_dir, cfg)

def worker(gpu_id, imgs, directory_initializer, cfg):
    torch.cuda.set_device(gpu_id)
    log.info(f"Process on GPU {gpu_id} started.")
    for image_path in imgs:
        json_path = str(directory_initializer.metadata_dir / f"{image_path.stem}.json")
        input_paths = (image_path, json_path)
        process_single_image(input_paths, directory_initializer.instance_mask_dir, directory_initializer.semantic_mask_dir, directory_initializer.vis_masks, cfg)

def process_concurrently(directory_initializer: DirectoryInitializer, cfg: DictConfig):
    imgs = directory_initializer.get_images()
    num_gpus = torch.cuda.device_count()
    num_workers = min(num_gpus, len(imgs))  # Use minimum of available GPUs or images
    chunk_size = math.ceil(len(imgs) / num_workers)
    log.info(f"Chunk size: {chunk_size}")
    chunks = [imgs[i:i + chunk_size] for i in range(0, len(imgs), chunk_size)]
    log.info(f"Chunk length: {len(chunks)}")
    processes = []
    for gpu_id, img_chunk in enumerate(chunks):
        if gpu_id >= num_workers:
            break
        p = mp.Process(target=worker, args=(gpu_id, img_chunk, directory_initializer, cfg))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def main(cfg: DictConfig) -> None:
    log.info("Starting batch processing")
    directory_initializer = DirectoryInitializer(cfg)
    if cfg.segment.sam.multiprocess:
        log.info("Starting concurrent processing")
        process_concurrently(directory_initializer, cfg)
    else:
        log.info("Starting sequential processing")
        processor = SingleImageProcessor(cfg, directory_initializer)
        process_sequentially(directory_initializer, processor, cfg)
