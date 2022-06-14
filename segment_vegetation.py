import logging
from dataclasses import asdict
from multiprocessing import Manager, Pool, Process, cpu_count
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from semif_utils.datasets import BatchMetadata, Cutout
from semif_utils.segment_utils import (ClassifyMask, GenCutoutProps,
                                       VegetationIndex, get_image_meta,
                                       get_watershed, prep_bbox,
                                       seperate_components, thresh_vi)
from semif_utils.utils import (apply_mask, clear_border, crop_cutouts,
                               dilate_erode, get_upload_datetime, make_exg,
                               reduce_holes)

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
        self.vi = cfg.segment.vi
        self.class_algorithm = cfg.segment.class_algorithm
        self.multi_process = cfg.segment.multiprocess

        if not self.cutout_batch_dir.exists():
            self.cutout_batch_dir.mkdir(parents=True, exist_ok=True)

        # self.cutout_pipeline()

    def process_domain(self, img):
        """First cutout processing step of the full image."""
        v_index = getattr(VegetationIndex(), self.vi)
        vi = v_index(img)
        th_vi = thresh_vi(vi)
        # Get classified mask
        clalgo = getattr(ClassifyMask(), self.class_algorithm)
        mask = clalgo(th_vi)
        return mask

    def process_cutout(self, cutout):
        """Second cutout processing step.
        """
        vi = make_exg(cutout, thresh=True)
        thresh_vi_arr = thresh_vi(vi, low=10, upper=100, sigma=5)

        # process the watershed
        wtrshed_lbls = get_watershed(thresh_vi_arr)

        mask = np.where(wtrshed_lbls <= 0.3, 0., 1)

        dil_ero_mask = dilate_erode(mask[:, :, 0],
                                    kernel_size=3,
                                    dil_iters=5,
                                    eros_iters=6,
                                    hole_fill=True)

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

        # for box in bboxes:
        for box in bboxes:

            # Scale the box that will be used for the cutout
            scale = [imgdata.fullres_width, imgdata.fullres_height]
            box, x1, y1, x2, y2 = prep_bbox(box, scale)

            # Crop image to bbox
            rgb_crop = rgb_array[y1:y2, x1:x2]
            mask = self.process_domain(rgb_crop)
            # Clear borders
            if self.clear_border:
                mask = clear_border(mask) * 255
            # Separate components
            list_cutouts_masks = seperate_components(mask)
            # Create RGB cutout for second round of processing
            cutout_0 = apply_mask(rgb_crop, mask, "black")
            # Second round of processing
            for cut_mask in list_cutouts_masks:
                preproc_cutout = apply_mask(cutout_0, cut_mask, "black")
                mask2 = self.process_cutout(preproc_cutout)

                new_cutout = apply_mask(preproc_cutout, mask2, "black")
                new_cropped_cutout = crop_cutouts(new_cutout)

                # Get regionprops
                cutprops = GenCutoutProps(mask2).to_dataclass()
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
                                is_primary=box.is_primary
                                #cutout_species=species_cls
                                )
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
    labels = [str(x) for x in (metadir).glob("*.json")]
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
        labels = [str(x) for x in (metadir).glob("*.json")]
        for label in labels:
            svg.cutout_pipeline(label)
        log.info("Finished segmenting vegetation")
