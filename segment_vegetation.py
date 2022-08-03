import logging
from dataclasses import asdict
from multiprocessing import Manager, Pool, Process, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from semif_utils.datasets import BatchMetadata, Cutout
from semif_utils.segment_algos import process_general,thresh_vi, reduce_holes
from semif_utils.segment_utils import (GenCutoutProps, SegmentMask,
                                       VegetationIndex, prep_bbox)
from semif_utils.utils import (apply_mask, clear_border, crop_cutouts,
                               dilate_erode, get_image_meta,
                               get_upload_datetime, get_watershed, make_exg)

log = logging.getLogger(__name__)


class SegmentVegetation:

    def __init__(self, cfg: DictConfig) -> None:

        self.data_dir = Path(cfg.data.datadir)
        self.cutout_dir = Path(cfg.data.cutoutdir)
        self.batchdir = Path(cfg.data.batchdir)
        self.batch_id = self.batchdir.name
        self.metadata = self.batchdir / "metadata"
        self.cutout_batch_dir = self.cutout_dir / self.batch_id
        self.clear_border = cfg.segment.clear_border

        self.dom_vi = cfg.segment.domain.vi
        self.dom_vi_th = cfg.segment.domain.vi_th
        self.dom_th_low = cfg.segment.domain.th_low
        self.dom_th_up = cfg.segment.domain.th_up
        self.dom_th_sig = cfg.segment.domain.th_sig
        self.dom_algo = cfg.segment.domain.class_algo
        self.dom_kern_size = cfg.segment.domain.kern_size
        self.dom_dil_iter = cfg.segment.domain.dil_iter
        self.dom_ero_iter = cfg.segment.domain.ero_iter
        self.dom_hole_fill = cfg.segment.domain.hole_fill

        self.cut_vi = cfg.segment.cutout.vi
        self.cut_vi_th = cfg.segment.cutout.vi_th
        self.cut_th_low = cfg.segment.cutout.th_low
        self.cut_th_up = cfg.segment.cutout.th_up
        self.cut_th_sig = cfg.segment.cutout.th_sig
        self.cut_algo = cfg.segment.cutout.class_algo
        self.cut_kern_size = cfg.segment.cutout.kern_size
        self.cut_dil_iter = cfg.segment.cutout.dil_iter
        self.cut_ero_iter = cfg.segment.cutout.ero_iter
        self.cut_hole_fill = cfg.segment.cutout.hole_fill

        self.multi_process = cfg.segment.multiprocess

        if not self.cutout_batch_dir.exists():
            self.cutout_batch_dir.mkdir(parents=True, exist_ok=True)

    def process_domain(self, img):
        """First cutout processing step of the full image."""
        v_index = getattr(VegetationIndex(), self.dom_vi)
        clalgo = getattr(SegmentMask(), self.dom_algo)
        mask = process_general(img, v_index, 0, clalgo)
        return mask

    def process_cutout(self, cutout):
        """Second cutout processing step.
        """
        v_index = getattr(VegetationIndex(), self.cut_vi)
        vi = v_index(cutout, thresh=self.cut_vi_th)
        th_vi = thresh_vi(vi,
                          low=self.cut_th_low,
                          upper=self.cut_th_up,
                          sigma=self.cut_th_sig)

        clalgo = getattr(SegmentMask(), self.dom_algo)
        mask = clalgo(th_vi)
        mask = np.where(mask <= 0.3, 0., 1)
        dil_ero_mask = dilate_erode(mask,
                                    kernel_size=self.cut_kern_size,
                                    dil_iters=self.cut_dil_iter,
                                    eros_iters=self.cut_ero_iter,
                                    hole_fill=self.cut_hole_fill)
        reduced_mask = reduce_holes(dil_ero_mask * 255) * 255
        return reduced_mask

    def cutout_pipeline(self, payload):
        """ Main Processing pipeline. Reads images from list of labels in
            labeldir,             
        """

        imgdata = payload["imgdata"] if self.multi_process else get_image_meta(
            payload)
        # Call image array
        rgb_array = imgdata.array

        # Get bboxes
        bboxes = imgdata.bboxes
        ## Process on images by individual bbox detection
        cutout_num = 0
        cutout_ids = []

        for box in bboxes:
            # Scale the box that will be used for the cutout
            scale = [imgdata.fullres_width, imgdata.fullres_height]
            box, x1, y1, x2, y2 = prep_bbox(box, scale)
            # Crop image to bbox
            rgb_crop = rgb_array[y1:y2, x1:x2]
            if rgb_crop.sum() == 0:
                continue
            mask = self.process_domain(rgb_crop)
            # Clear borders
            if self.clear_border:
                mask = clear_border(mask) * 255
            if mask.max() == 0:
                continue
            # Separate components
            # list_cutouts_masks = seperate_components(mask)
            # Create RGB cutout for second round of processing
            cutout_0 = apply_mask(rgb_crop, mask, "black")
            # Second round of processing
            # for cut_mask in list_cutouts_masks:
            #     preproc_cutout = apply_mask(cutout_0, cut_mask, "black")
                # mask2 = self.process_cutout(preproc_cutout)

            # new_cutout = apply_mask(preproc_cutout, mask2, "black")
            new_cropped_cutout = crop_cutouts(cutout_0)

            # Get regionprops
            if np.sum(mask == 0) == mask.shape[0] * mask.shape[1]:
                continue
            cutprops = GenCutoutProps(mask).to_dataclass()
            # Removes false positives that are typically very small cutouts
            if type(cutprops.area) is not list and cutprops.area < 500:
                continue

            # Create dataclass
            cutout = Cutout(blob_home=self.data_dir.name,
                            data_root=self.cutout_dir.name,
                            batch_id=self.batch_id,
                            image_id=imgdata.image_id,
                            cutout_num=cutout_num,
                            datetime=imgdata.exif_meta.DateTime,
                            cutout_props=asdict(cutprops),
                            is_primary=box.is_primary,
                            cls=box.cls)
            cutout.save_cutout(new_cropped_cutout)
            cutout.save_config(self.cutout_dir)

            cutout_ids.append(cutout.cutout_id)
            cutout_num += 1

        # To json
        imgdata.cutout_ids = cutout_ids
        imgdata.save_config(self.metadata)


def return_dataclass_list(label, return_lbls):
    dc = get_image_meta(label)
    return_lbls.append(dc)
    return return_lbls


def create_dataclasses(metadir):
    log.info("Creating dataclasses")
    labels = sorted([str(x) for x in (metadir).glob("*.json")])
    jobs = []
    manager = Manager()
    return_list = manager.list()
    for lbl in labels:
        p = Process(target=return_dataclass_list, args=(lbl, return_list))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    log.info(f"Converted metadata to {len(return_list)} dataclasses")
    return return_list


def main(cfg: DictConfig) -> None:
    # Create batch metadata
    data_dir = Path(cfg.data.datadir)
    data_root = Path(cfg.data.developeddir)
    batch_dir = Path(cfg.data.batchdir)
    upload_datetime = get_upload_datetime(cfg.data.batchdir)

    BatchMetadata(blob_home=data_dir.name,
                  data_root=data_root.name,
                  batch_id=batch_dir.name,
                  upload_datetime=upload_datetime).save_config()

    svg = SegmentVegetation(cfg)

    # Multiprocessing
    metadir = Path(f"{cfg.data.batchdir}/metadata")
    if cfg.segment.multiprocess:
        return_list = create_dataclasses(metadir)
        payloads = []
        for idx, img in enumerate(return_list):
            data = {"imgdata": img, "idx": idx}
            payloads.append(data)

        log.info("Multi-Processing image data.")
        procs = cpu_count()
        pool = Pool(processes=procs)
        pool.map(svg.cutout_pipeline, payloads)
        pool.close()
        pool.join()
        log.info("Finished segmenting vegetation")
    else:
        # Single process
        log.info("Processing image data.")
        labels = sorted([str(x) for x in (metadir).glob("*.json")])
        for label in labels:
            svg.cutout_pipeline(label)
        log.info("Finished segmenting vegetation")
