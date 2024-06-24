import copy
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from semif_utils.datasets import BatchMetadata, Cutout, SegmentData
from semif_utils.segment_species import Segment
from semif_utils.segment_utils import (GenCutoutProps, generate_new_color,
                                       prep_bbox)
from semif_utils.utils import (apply_mask, create_dataclasses, cutoutmeta2csv,
                               get_upload_datetime, reduce_holes, save_json)
from tqdm import tqdm

log = logging.getLogger(__name__)


class SegmentVegetation:
    def __init__(self, cfg: DictConfig) -> None:
        self.data_name = Path(cfg.data.datadir).name
        self.cutout_dir = Path(cfg.data.cutoutdir)
        self.batchdir = Path(cfg.data.batchdir)
        self.season = cfg.general.season
        self.batch_id = self.batchdir.name
        self.species_path = cfg.data.species
        self.metadata = cfg.batchdata.metadata
        self.mtshp_project_path = Path(
            self.batchdir, "autosfm", "project", f"{self.batch_id}.psx"
        )
        # Create image mask directories
        self.binary_mask_dir = Path(cfg.batchdata.meta_masks, "binary_masks")
        self.semantic_mask_dir = Path(cfg.batchdata.meta_masks, "semantic_masks")
        self.instance_mask_dir = Path(cfg.batchdata.meta_masks, "instance_masks")

        self.cutout_batch_dir = self.cutout_dir / self.batch_id
        self.multi_process = cfg.segment.multiprocess

        if not self.cutout_batch_dir.exists():
            self.cutout_batch_dir.mkdir(parents=True, exist_ok=True)
        # if not self.binary_mask_dir.exists():
        #     self.binary_mask_dir.mkdir(parents=True, exist_ok=True)
        if not self.semantic_mask_dir.exists():
            self.semantic_mask_dir.mkdir(parents=True, exist_ok=True)
        if not self.instance_mask_dir.exists():
            self.instance_mask_dir.mkdir(parents=True, exist_ok=True)

        self.bbox_areas = []
        self.bbox_th = None
        self.coty_th = None  # 75th percentile mark

    def bboxareas(self, imagedata_list):
        """Calculates the mean and the 25% interquartile of the area
        of all bboxes for all images."""
        # TODO calculate average area for each species and store in dictionary
        # key = species, value = average area
        scale = [imagedata_list[0].fullres_width, imagedata_list[0].fullres_height]
        per_img_bboxes = [img.bboxes for img in imagedata_list]
        boxs = [box for box in per_img_bboxes]
        for box in boxs:
            for bo in box:
                box, x1, y1, x2, y2 = prep_bbox(bo, scale, save_changes=False)
                width = float(x2) - float(x1)
                length = float(y2) - float(y1)
                area = width * length
                self.bbox_areas.append(area)
        self.bbox_th = mean(self.bbox_areas)
        self.coty_th = pd.Series(self.bbox_areas).describe().iloc[4]

    def config_from_bboxarea(self, boxarea):
        if boxarea < 25000:  # Very very small
            min_object_size = 100
            min_hole_size = 100
            median_kernel = 1
        elif boxarea < 50000:  # Very small
            min_object_size = 500
            min_hole_size = 500
            median_kernel = 3

        elif boxarea < 100000:  # Small
            min_object_size = 1000
            min_hole_size = 1000
            median_kernel = 7

        elif boxarea < 200000:  # Med
            min_object_size = 5000
            min_hole_size = 5000
            median_kernel = 9

        elif boxarea >= 200000:  # Large
            min_object_size = 10000
            min_hole_size = 10000
            median_kernel = 11
        return min_object_size, min_hole_size, median_kernel

    def create_instance_mask(self, instance_mask, instance_palette, box_instance_id):
        color_mask = np.zeros(
            (instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8
        )
        # palette = np.array(instance_palette).transpose()

        # all(2) force all channels to be equal
        # any(-1) matches any color
        # temp_mask = (instance_mask[:, :, :, None] == palette).all(2).any(-1)
        for label_id in np.unique(instance_palette):
            if label_id == 0:
                continue  # Skip background
            if label_id != box_instance_id:
                color_mask[instance_mask == label_id] = box_instance_id

        # target color
        # rgb_arr = np.array(box_instance_id)

        # instance_mask = np.where(temp_mask[:, :, None], instance_mask,
        #  rgb_arr[None, None, :])

        return color_mask

    def create_semantic_mask(self, semantic_mask, box_instance_id):
        semantic_mask[semantic_mask == 255] = box_instance_id

        return semantic_mask

    def cutout_pipeline(self, payload):
        """Main Processing pipeline. Reads images from list of labels in
        labeldir,
        """
        imgdata = payload["imgdata"] if self.multi_process else payload
        log.info(f"Processing {imgdata.image_id}")
        # Call image array
        rgb_array = imgdata.array

        # Get bboxes
        bboxes = imgdata.bboxes

        # Process on images by individual bbox detection

        cutout_ids = []

        semantic_mask_zeros = np.zeros(rgb_array.shape[:2], dtype=np.float32)
        instance_mask_zeros = np.zeros(rgb_array.shape, dtype=np.float32)

        log.debug(
            f"Processing {len(bboxes)} bounding boxes for image {imgdata.image_id}."
        )
        instance_colors = [[0, 0, 0]]
        instance_palette = [[0, 0, 0]]
        semantic_palette = [[0]]
        for box in bboxes:
            # Scale the box that will be used for the cutout
            spec_cls = box.cls
            cm_name = spec_cls["common_name"]
            scale = [imgdata.fullres_width, imgdata.fullres_height]
            box, x1, y1, x2, y2 = prep_bbox(box, scale)

            if any(val < 0 for val in (y1, y2, x1, x2)):
                log.debug(
                    f"Some coordinates values are negative: {y1}, {y2}, {x1}, {x2}"
                )
                coords = tuple(max(0, val) for val in (y1, y2, x1, x2))
                y1, y2, x1, x2 = coords[0], coords[1], coords[2], coords[3]
                log.debug(f"Transformed coordinates to: {y1}, {y2}, {x1}, {x2}")
                # continue

            rgb_crop = rgb_array[y1:y2, x1:x2]

            data = SegmentData(
                species=spec_cls,
                species_info=self.species_path,
                bbox=(x1, y1, x2, y2),
                bbox_size_th=self.bbox_th,
            )

            seg = Segment(rgb_crop, data)
            boxarea = seg.get_bboxarea()

            if boxarea > 15000000:
                log.debug(
                    f"\n*******************\n Likely multiple plants in a single detection results. \n{imgdata.image_id} - {boxarea} \nPrimary status: {box.is_primary} \n*************\n"
                )

            if boxarea < 25000:
                log.debug("Processing cotyldon.")
                seg.mask = seg.cotlydon()

            else:
                # TODO: make multiple options for based on species
                log.debug(f"Processing vegetative plant of size {boxarea}.")
                seg.mask = seg.general_seg(mode="cluster")

            if seg.is_mask_empty() and cm_name != "colorchecker":
                log.debug("Mask is empty and it's not a color checker. Ignoring cutout")
                continue

            if boxarea > 200000:
                log.debug("Processing Round 2 for large plant")
                seg = Segment(apply_mask(rgb_crop, seg.mask, "black"), data)
                # General segmentation
                seg.mask = seg.general_seg(mode="cluster")
            # Identify hole filling sizes and kernel for leaf edge smoothing based on bbox area
            min_object_size, min_hole_size, median_kernel = self.config_from_bboxarea(
                boxarea
            )

            # Reduce holes
            if seg.rem_bbotblue():
                log.debug("Removing benchbot blue from images")
                seg.mask = (
                    reduce_holes(
                        seg.mask,
                        min_object_size=min_object_size,
                        min_hole_size=min_hole_size,
                    ).astype(np.uint8)
                    * 255
                )
            # Smooth mask edges
            log.debug("Smoothing edges")
            seg.mask = cv2.medianBlur(seg.mask.astype(np.uint8), median_kernel)

            if seg.is_mask_empty() and cm_name != "colorchecker":
                log.debug(
                    "Mask is empty and it's not a color checker (2). Ignoring cutout."
                )
                continue

            # Get morphological properties
            log.debug("Calculating cutout properties.")
            seg.props = GenCutoutProps(rgb_crop, seg.mask).to_dataclass()
            # Skip saving non-green cutouts unless its a colorchecker result
            if seg.props.green_sum < 5 and cm_name != "colorchecker":
                log.debug(
                    "Cutout is not green and it's not a color checker. Ignoring cutout."
                )
                continue

            # Prep box by removing some redundant records
            is_primary = box.is_primary
            box.instance_rgb_id = generate_new_color(instance_colors, pastel_factor=0.7)

            box.cutout_exists = True
            boxdict = asdict(box)
            boxdict.pop("cls")
            boxdict.pop("is_primary")
            # Create dataclass
            log.debug("Creating cutout dataclass.")
            cutout = Cutout(
                data_root=self.cutout_dir.name,
                season=self.season,
                batch_id=self.batch_id,
                image_id=imgdata.image_id,
                bbox=boxdict,
                cutout_id=box.bbox_id,
                datetime=imgdata.exif_meta.DateTime,
                cutout_props=asdict(seg.props),
                is_primary=is_primary,
                hwc=rgb_crop.shape,
                cls=seg.species,
                camera_info=imgdata.camera_info,
                exif_meta=imgdata.exif_meta,
                extends_border=seg.get_extends_borders(seg.mask),
            )
            cutout_dir = self.cutout_dir
            cutout_array = apply_mask(rgb_crop, seg.mask, "black")
            cutout_mask = np.zeros(seg.mask.shape[:2])
            cutout_mask[seg.mask != 0] = box.cls["class_id"]
            log.debug("Saving cutout data.")
            cutout.save_cutout(cutout_dir, cutout_array)
            cutout.save_config(cutout_dir)
            cutout.save_cropout(cutout_dir, rgb_crop)
            cutout.save_cutout_mask(cutout_dir, cutout_mask)
            cutout_ids.append(cutout.cutout_id)

            ####################################
            ## Create masks only for non-colorchecker items
            if cm_name == "colorchecker":
                continue
            # Create semantic mask
            log.debug("Mapping colors to the semantic mask.")
            semantic_mask_zeros[y1:y2, x1:x2] = cutout_mask
            semantic_palette.append(box.cls["class_id"])
            # Create instance mask
            log.debug("Mapping colors to the instance mask.")
            # Assign unique colors to each labeled component
            instance_mask_zeros[y1:y2, x1:x2][seg.mask == 255] = box.instance_rgb_id
            instance_colors.append(box.instance_rgb_id)
            instance_palette.append(box.instance_rgb_id)

            ####################################

        log.debug(f"Saved {len(cutout_ids)} cutouts for image {imgdata.image_id}")
        # To json
        imgdata.cutout_ids = cutout_ids
        imgdata.save_config(self.metadata)
        imgdata.save_mask(self.semantic_mask_dir, semantic_mask_zeros)
        imgdata.save_mask(self.instance_mask_dir, instance_mask_zeros)
        log.debug(f"{imgdata.image_id} finished.")


def main(cfg: DictConfig) -> None:
    start = time.time()
    # Create batch metadata
    data_dir = Path(cfg.data.datadir)
    data_root = Path(cfg.data.developeddir)
    batch_dir = Path(cfg.data.batchdir)
    upload_datetime = get_upload_datetime(cfg.data.batchdir)
    cutoutdir = cfg.data.cutoutdir
    batch_id = cfg.general.batch_id
    csv_path = Path(cutoutdir, batch_id, f"{batch_id}.csv")

    # Create and save batchmetadata json
    img_list = [x.name for x in Path(cfg.batchdata.images).glob("*.jpg")]
    bm = BatchMetadata(  # blob_home=data_dir.name,
        data_root=data_root.name,
        batch_id=batch_dir.name,
        upload_datetime=upload_datetime,
        image_list=img_list,
    )
    save_json(cfg.batchdata.batchmeta, asdict(bm))

    svg = SegmentVegetation(cfg)

    # Multiprocessing
    metadir = Path(f"{cfg.data.batchdir}/metadata")
    if cfg.segment.multiprocess:
        return_list = create_dataclasses(metadir, cfg)
        # Get all bbox areas
        svg.bboxareas(return_list)
        payloads = []
        for idx, img in enumerate(return_list):
            data = {"imgdata": img, "idx": idx}
            payloads.append(data)

        log.info(f"Multi-Processing image data for batch {batch_dir.name}.")
        procs = int(len(os.sched_getaffinity(0)) / 6)
        with ProcessPoolExecutor(max_workers=procs) as executor:
            # Submit tasks using map
            executor.map(svg.cutout_pipeline, payloads)
        log.info(f"Finished segmenting vegetation for batch {batch_dir.name}")
    else:
        # Single process
        log.info(f"Processing image data for batch {batch_dir.name}.")
        return_list = create_dataclasses(metadir, cfg)

        svg.bboxareas(return_list)
        for imgdata in tqdm(return_list):
            svg.cutout_pipeline(imgdata)
        log.info(f"Finished segmenting vegetation for batch {batch_dir.name}")

    log.info(f"Condensing cutout results into a single csv file.")
    df = cutoutmeta2csv(cutoutdir, batch_id, csv_path, save_df=True)

    end = time.time()
    log.info(f"Segmentation completed in {end - start} seconds.")
    log.info(f"{len(df)} cutouts created.")
