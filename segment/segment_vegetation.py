import logging
import time
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from semif_utils.cutout_contours import mask_to_polygons
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
        self.mtshp_project_path = Path(self.batchdir, "autosfm", "project",
                                       f"{self.batch_id}.psx")
        # Create image mask directories
        self.binary_mask_dir = Path(cfg.batchdata.meta_masks, "binary_masks")
        self.semantic_mask_dir = Path(cfg.batchdata.meta_masks,
                                      "semantic_masks")
        self.instance_mask_dir = Path(cfg.batchdata.meta_masks,
                                      "instance_masks")

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
        """ Calculates the mean and the 25% interquartile of the area 
            of all bboxes for all images. """
        # TODO calculate average area for each species and store in dictionary
        # key = species, value = average area
        scale = [
            imagedata_list[0].fullres_width, imagedata_list[0].fullres_height
        ]
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

    def cutout_pipeline(self, payload):
        """ Main Processing pipeline. Reads images from list of labels in
            labeldir,             
        """
        imgdata = payload["imgdata"] if self.multi_process else payload
        # Call image array
        rgb_array = imgdata.array

        # Get bboxes
        bboxes = imgdata.bboxes

        ## Process on images by individual bbox detection
        cutout_num = 0
        cutout_ids = []
        semantic_mask_zeros = np.zeros(rgb_array.shape, dtype=np.uint8)
        instance_mask_zeros = np.zeros(rgb_array.shape, dtype=np.uint8)
        colors = []
        log.info(
            f"Processing {len(bboxes)} bounding boxes for image {imgdata.image_id}."
        )
        for box in bboxes:
            # Scale the box that will be used for the cutout
            spec_cls = box.cls
            cm_name = spec_cls["common_name"]

            scale = [imgdata.fullres_width, imgdata.fullres_height]
            box, x1, y1, x2, y2 = prep_bbox(box, scale)

            rgb_crop = rgb_array[y1:y2, x1:x2]

            if rgb_crop.size <= 0:
                print(y1, y2, x1, x2)
                log.warning("rgb crop size <= 0.")
                continue

            data = SegmentData(species=spec_cls,
                               species_info=self.species_path,
                               bbox=(x1, y1, x2, y2),
                               bbox_size_th=self.bbox_th)

            seg = Segment(rgb_crop, data)
            boxarea = seg.get_bboxarea()

            if boxarea < self.bbox_th and boxarea < 25000:
                log.info("Processing cotyldon.")
                seg.mask = seg.cotlydon()

            else:
                # TODO: make multiple options for based on species
                seg.mask = seg.general_seg(mode="cluster")

            if seg.is_mask_empty() and cm_name != "colorchecker":
                log.warning(
                    "Mask is empty and it's not a color checker. Ignoring cutout"
                )
                continue

            # seg.props = GenCutoutProps(rgb_crop, seg.mask).to_dataclass()

            if boxarea < 200000 == 2:
                # log.info("Round 2")
                seg = Segment(apply_mask(rgb_crop, seg.mask, "black"), data)
                # General segmentation
                seg.mask = seg.general_seg(mode="cluster")

            if seg.rem_bbotblue():
                seg.mask = reduce_holes(seg.mask,
                                        min_object_size=10000,
                                        min_hole_size=10000)

            # Identify kernel for leaf edge smoothing based on bbox area
            if boxarea < 25000:  # Very very small
                median_kernel = 1
            elif boxarea < 50000:  # Very small
                median_kernel = 3

            elif boxarea < 100000:  # Small
                median_kernel = 7

            elif boxarea < 200000:  # Med
                median_kernel = 9

            elif boxarea > 200000:  # Large
                median_kernel = 11

            # Smooth mask edges
            seg.mask = cv2.medianBlur(seg.mask.astype(np.uint8), median_kernel)

            if seg.is_mask_empty() and cm_name != "colorchecker":
                log.warning(
                    "Mask is empty and it's not a color checker (2). Ignoring cutout."
                )
                continue
            seg.mask = np.where(seg.mask != 0, 1, 0)

            # Create semantic mask
            seg.mask = np.where(seg.mask != 0, 255, 0)
            semantic_mask_zeros[y1:y2, x1:x2][seg.mask == 255] = seg.class_id
            # Create instance mask
            box.instance_id = generate_new_color(colors, pastel_factor=0.7)
            colors.append(box.instance_id)
            instance_mask_zeros[y1:y2,
                                x1:x2][seg.mask == 255] = box.instance_id

            cutout_array = apply_mask(rgb_crop, seg.mask, "black")

            # Get morphological properties
            seg.props = GenCutoutProps(rgb_crop, seg.mask).to_dataclass()

            # Skip saving non-green cutouts unless its a colorchecker result
            if seg.props.green_sum < 5 and cm_name != "colorchecker":
                log.warning(
                    "Cutout is not green and it's not a color checker. Ignoring cutout."
                )
                continue
            # calculate mean of cutout and rgb_crops
            # rgb_cropout_mean = calc_ch_means(rgb_crop)
            # rgb_cutout_mean = calc_ch_means(cutout_array, cutout=True)

            # cropped_mask2 = crop_cutouts(seg.mask)
            # cutout_contours = mask_to_polygons(seg.mask,
            #                                    epsilon=10.,
            #                                    min_area=10)

            # cropped_cutout2 = crop_cutouts(cutout_array)

            # Create dataclass
            cutout = Cutout(
                blob_home=self.data_name,
                data_root=self.cutout_dir.name,
                season=self.season,
                batch_id=self.batch_id,
                image_id=imgdata.image_id,
                cutout_num=cutout_num,
                datetime=imgdata.exif_meta.DateTime,
                cutout_props=asdict(seg.props),
                # rgb_cropout_mean=rgb_cropout_mean,
                # rgb_cutout_mean=rgb_cutout_mean,
                # local_contours=cutout_contours,
                is_primary=box.is_primary,
                cls=seg.species,
                extends_border=seg.get_extends_borders())

            # if very_small == 1:
            # cutout.save_verysmall_cropout(rgb_crop, boxarea)
            cutout_dir = self.cutout_dir
            cutout.save_cutout(cutout_dir, cutout_array)
            cutout.save_config(cutout_dir)
            cutout.save_cropout(cutout_dir, rgb_crop)
            cutout.save_cutout_mask(cutout_dir,
                                    seg.mask.astype(np.uint8) * 255)
            cutout_ids.append(cutout.cutout_id)
            cutout_num += 1
        log.info(
            f"Saved {len(cutout_ids)} cutouts for image {imgdata.image_id}")
        # To json
        imgdata.cutout_ids = cutout_ids
        imgdata.save_config(self.metadata)
        # imgdata.save_binary_mask(binary_mask_zeros)
        imgdata.save_semantic_mask(self.semantic_mask_dir, semantic_mask_zeros)
        imgdata.save_instance_mask(self.instance_mask_dir, instance_mask_zeros)
        log.info(f"{imgdata.image_id} finished.")


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
    bm = BatchMetadata(blob_home=data_dir.name,
                       data_root=data_root.name,
                       batch_id=batch_dir.name,
                       upload_datetime=upload_datetime,
                       image_list=img_list)
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
        procs = cpu_count() - cfg.segment.cpus_left
        with Pool(processes=procs) as p:
            p.imap_unordered(svg.cutout_pipeline, payloads)
            p.close()
            p.join()
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
    cutoutmeta2csv(cutoutdir, batch_id, csv_path, save_df=True)

    end = time.time()
    log.info(f"Segmentation completed in {end - start} seconds.")