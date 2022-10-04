import logging
from dataclasses import asdict
from multiprocessing import Manager, Pool, Process, cpu_count
from pathlib import Path
from semif_utils.segment_species import Segment
import numpy as np
from omegaconf import DictConfig

from semif_utils.segment_species import Segment
from semif_utils.cutout_contours import CutoutMapper, mask_to_polygons
from semif_utils.datasets import BatchMetadata, Cutout, SegmentData
from semif_utils.segment_utils import GenCutoutProps, prep_bbox, get_species_info
from semif_utils.utils import (apply_mask, crop_cutouts, get_image_meta,
                               get_upload_datetime)
from statistics import mean
import pandas as pd
log = logging.getLogger(__name__)


class SegmentVegetation:
    def __init__(self, cfg: DictConfig) -> None:

        self.data_dir = Path(cfg.data.datadir)
        self.cutout_dir = Path(cfg.data.cutoutdir)
        self.batchdir = Path(cfg.data.batchdir)
        self.batch_id = self.batchdir.name
        self.species_path = cfg.data.species
        self.metadata = self.batchdir / "metadata"
        self.mtshp_project_path = Path(self.batchdir, "autosfm", "project",
                                       f"{self.batch_id}.psx")
        self.binary_mask_dir = self.batchdir / "binary_masks"
        self.semantic_mask_dir = self.batchdir / "semantic_masks"
        self.instance_mask_dir = self.batchdir / "instance_masks"

        self.cutout_batch_dir = self.cutout_dir / self.batch_id
        self.clear_border = cfg.segment.clear_border
        self.multi_process = cfg.segment.multiprocess

        if not self.cutout_batch_dir.exists():
            self.cutout_batch_dir.mkdir(parents=True, exist_ok=True)
        if not self.binary_mask_dir.exists():
            self.binary_mask_dir.mkdir(parents=True, exist_ok=True)
        if not self.semantic_mask_dir.exists():
            self.semantic_mask_dir.mkdir(parents=True, exist_ok=True)
        if not self.instance_mask_dir.exists():
            self.instance_mask_dir.mkdir(parents=True, exist_ok=True)
        
        self.bbox_areas = []
        self.bbox_th = None
        self.sevenfive = None # 75th percentile mark
    
    def bboxareas(self, imagedata_list):
        scale = [imagedata_list[0].fullres_width, imagedata_list[0].fullres_height]
        per_img_bboxes = [img.bboxes for img in imagedata_list]
        boxs = [box for box in per_img_bboxes]      
        for box in boxs:
            for bo in box:
                box, x1, y1, x2, y2 = prep_bbox(bo, scale)
                width = float(x2) - float(x1)
                length = float(y2) - float(y1)
                area = width * length
                self.bbox_areas.append(area)
        self.bbox_th = mean(self.bbox_areas)
        self.sevenfive = pd.Series(self.bbox_areas).describe().iloc[6]
         
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
        binary_mask_zeros = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
        for box in bboxes:
            # Scale the box that will be used for the cutout
            spec = box.cls
            scale = [imgdata.fullres_width, imgdata.fullres_height]
            box, x1, y1, x2, y2 = prep_bbox(box, scale)
            rgb_crop = rgb_array[y1:y2, x1:x2]
            species_info = get_species_info(self.species_path, spec['USDA_symbol'])
            rgb = species_info['rgb']
            hex = species_info['hex']

            data = SegmentData(species=box.cls,
                               growth_stage=imgdata.growth_stage,
                               bbox=(x1, y1, x2, y2),
                               bbox_size_th=self.bbox_th)

            seg = Segment(rgb_crop, data)
            boxarea = seg.get_bboxarea()
            
            if boxarea < self.sevenfive and "coty" not in seg.growth_stage:
                log.info("Very small detection result. Ignoring.")
                break
            
            if boxarea < self.bbox_th and "coty" not in seg.growth_stage:
                log.info("Small detection result. Ignoring.")
                break

            if boxarea < self.bbox_th and "coty" in seg.growth_stage:
                log.info("Processing coty.")
                seg.mask = seg.cotlydon()
        
            else:
                log.info("Processing vegetative")
                # TODO: make multiple options for based on species
                # seg.mask = seg.multi_otsu(img_type="vi", classes=3)
                # seg.mask = seg.lambsquarters(cotlydon=False)
                seg.mask = seg.general_seg(mode="cluster")

            if seg.is_mask_empty():
                break

            seg.props = GenCutoutProps(rgb_crop, seg.mask).to_dataclass()

            # if not seg.is_green():
            #     break
            
            if seg.rounds == 2:
                log.info("Round 2")
                seg = Segment(apply_mask(rgb_crop, seg.mask, "black"), data)
                # General segmentation
                seg.mask = seg.general_seg(mode="cluster")
                seg.props = GenCutoutProps(rgb_crop, seg.mask).to_dataclass()

                if seg.is_mask_empty():
                    break
            if 'seg.mask' in locals():
                binary_mask_zeros[y1:y2, x1:x2] = seg.mask

            cutout_array = apply_mask(rgb_crop, seg.mask, "black")
            cropped_mask2 = crop_cutouts(seg.mask)
            cutout_contours = mask_to_polygons(cropped_mask2,
                                               epsilon=10.,
                                               min_area=10)

            cropped_cutout2 = crop_cutouts(cutout_array)

            # Create dataclass
            cutout = Cutout(blob_home=self.data_dir.name,
                            data_root=self.cutout_dir.name,
                            batch_id=self.batch_id,
                            image_id=imgdata.image_id,
                            cutout_num=cutout_num,
                            datetime=imgdata.exif_meta.DateTime,
                            cutout_props=asdict(seg.props),
                            local_contours=cutout_contours,
                            is_primary=box.is_primary,
                            cls=seg.species,
                            extends_border=seg.get_extends_borders())
            cutout.save_cutout(cropped_cutout2)
            cutout.save_config(self.cutout_dir)

            cutout_ids.append(cutout.cutout_id)
            cutout_num += 1
            
        # To json
        imgdata.cutout_ids = cutout_ids
        imgdata.save_config(self.metadata)
        imgdata.save_binary_mask(binary_mask_zeros)


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
        # Get all bbox areas
        svg.bboxareas(return_list)
        payloads = []
        for idx, img in enumerate(return_list):
            data = {"imgdata": img, "idx": idx}
            payloads.append(data)

        log.info("Multi-Processing image data.")
        procs = cpu_count() - 1
        pool = Pool(processes=procs)
        pool.map(svg.cutout_pipeline, payloads)
        pool.close()
        pool.join()
        log.info("Finished segmenting vegetation")
    else:
        # Single process
        log.info("Processing image data.")
        return_list = create_dataclasses(metadir)
        svg.bboxareas(return_list)
        for imgdata in return_list:
            svg.cutout_pipeline(imgdata)
        log.info("Finished segmenting vegetation")
