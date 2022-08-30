import logging
from dataclasses import asdict
from multiprocessing import Manager, Pool, Process, cpu_count
from pathlib import Path
from segment_species import Segment
import cv2
import numpy as np
from omegaconf import DictConfig

from segment_species import Segment
from semif_utils.cutout_contours import CutoutMapper, mask_to_polygons
from semif_utils.datasets import BatchMetadata, Cutout
from semif_utils.segment_utils import GenCutoutProps, prep_bbox
from semif_utils.utils import (apply_mask, clear_border, crop_cutouts,
                               get_image_meta, get_upload_datetime)

log = logging.getLogger(__name__)

class SegmentVegetation:

    def __init__(self, cfg: DictConfig) -> None:

        self.data_dir = Path(cfg.data.datadir)
        self.cutout_dir = Path(cfg.data.cutoutdir)
        self.batchdir = Path(cfg.data.batchdir)
        self.batch_id = self.batchdir.name
        self.metadata = self.batchdir / "metadata"
        self.mtshp_project_path = Path(self.batchdir, "autosfm", "project", f"{self.batch_id}.psx")
        
        self.cutout_batch_dir = self.cutout_dir / self.batch_id
        self.clear_border = cfg.segment.clear_border
        self.multi_process = cfg.segment.multiprocess

        if not self.cutout_batch_dir.exists():
            self.cutout_batch_dir.mkdir(parents=True, exist_ok=True)

    def cutout_pipeline(self, payload):
        """ Main Processing pipeline. Reads images from list of labels in
            labeldir,             
        """
        self.cutout_mapper = CutoutMapper(self.mtshp_project_path)
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
            seg = Segment(rgb_crop)
            if rgb_crop.sum() == 0:
                log.info(f"Image crop sum of values = 0; {imgdata.image_path} ignored")
                continue
            species = box.cls
            g_stage = imgdata.growth_stage
            # if species["USDA_symbol"] == "CHAL7":
                # mask = seg.lambsquarters(cotlydon=False)
                # log.info("---Getting Region properties---")
            mask = seg.lambsquarters(cotlydon=False)
            extends_border = False if np.array_equal(mask, clear_border(mask)) else True
            if mask.max() == 0:
                continue

            cutout_0 = apply_mask(rgb_crop, mask, "black")

            mask2 = Segment(cutout_0).general_seg(mode="cluster")
            if mask2.max() == 0:
                continue
            props = GenCutoutProps(rgb_crop, mask2) 
            cutprops = props.to_dataclass()
            # Removes false positives that are typically very small cutouts
            if type(cutprops.area) is not list:
                if  g_stage != "cotyledon" and cutprops.area < 300:
                    continue
            if props.green_sum < 5000:
                continue
            cutout_2 = apply_mask(rgb_crop, mask2, "black")
            cropped_mask2 = crop_cutouts(mask2)
            cutout_contours = mask_to_polygons(cropped_mask2, epsilon=10., min_area=10, to_list=True)
            global_contours = self.cutout_mapper.map(cutout_contours, imgdata.image_id)
            cropped_cutout2 = crop_cutouts(cutout_2)
            # Create dataclass
            cutout = Cutout(blob_home=self.data_dir.name,
                            data_root=self.cutout_dir.name,
                            batch_id=self.batch_id,
                            image_id=imgdata.image_id,
                            cutout_num=cutout_num,
                            datetime=imgdata.exif_meta.DateTime,
                            cutout_props=asdict(cutprops),
                            local_contours=cutout_contours,
                            global_contours=global_contours,
                            is_primary=box.is_primary,
                            cls=box.cls,
                            extends_border=extends_border)
            cutout.save_cutout(cropped_cutout2)
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
