import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from scipy import ndimage as ndi
from skimage import measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import rank
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import watershed
from tqdm import tqdm

sys.path.append("..")

from datasets import (CUTOUT_PROPS, BatchConfigImages, BatchMetadata, Cutout,
                      ImageData)
from mongo_utils import Connect
from semif_utils import (apply_mask, clear_border, crop_cutouts, dilate_erode,
                         get_bbox_info, get_site_id, get_upload_datetime,
                         make_exg, make_exg_minus_exr, make_exr, make_kmeans,
                         make_ndi, otsu_thresh, reduce_holes)


class VegetationIndex:

    def exg(self, img):
        exg_vi = make_exg(img, thresh=True)
        return exg_vi

    def exr(self, img):
        exr_vi = make_exr(img)
        return exr_vi

    def exg_minus_exr(self, img):
        gmr_vi = make_exg_minus_exr(img)
        return gmr_vi

    def ndi(self, img):
        ndi_vi = make_ndi(img)
        return ndi_vi


class ClassifyMask:

    def otsu(self, vi):
        # Otsu's thresh
        vi_mask = otsu_thresh(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255
        return reduce_holes_mask

    def kmeans(self, vi):
        vi_mask = make_kmeans(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255

        return reduce_holes_mask


class SegmentVegetation:

    def __init__(self, bcfg, db, cfg: DictConfig) -> None:
        self.bcfg = bcfg
        self.batch_id = bcfg.batch_id
        self.batchdir = Path(cfg.general.batchdir)
        self.detectioncsv = self.batchdir / "detections.csv"  #bcfg.detections_csv
        self.clear_border = cfg.segment.clear_border  #bcfg.clear_border
        self.labels = (self.batchdir / "labels").glob("*.json")
        self.imagedir = Path(cfg.general.imagedir)
        states = ['TX', 'NC', 'MD']

        self.sitedir = [
            p for st in states for p in self.imagedir.parts if st in p
        ][0]

        self.date = self.sitedir.split("_")[-1]
        self.site_id = [st for st in states if st in self.sitedir][0]

        self.vi = bcfg.vi
        self.class_algorithm = bcfg.class_algorithm

        self.db = db
        self.processing_pipeline()

    def save_cutout(self, cutout, imgpath, cutout_num):
        cutout_dir = Path(self.imagedir, "cutouts")
        cutout_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{imgpath.stem}_{cutout_num}.png"
        cutout_path = cutout_dir / fname
        # return cutout_path
        cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
        return cutout_path

    def reprocess_bbox_cutouts(self, cutout):
        vi = make_exg(cutout, thresh=True)
        thresh_vi = np.where(vi <= 0, 0, vi)
        thresh_vi = np.where((thresh_vi > 10) & (thresh_vi < 100),
                             thresh_vi * 5, thresh_vi)
        # thresh_vi = np.where((thresh_vi > 70) & (thresh_vi < 100),
        #  thresh_vi * 2, thresh_vi)
        # kmeans = make_kmeans(thresh_vi)
        # reduced_mask = reduce_holes(kmeans * 255) * 255

        markers = rank.gradient(thresh_vi, disk(1)) < 12
        markers = ndi.label(markers)[0]
        gradient = rank.gradient(thresh_vi, disk(10))
        # process the watershed
        labels = watershed(gradient, markers)
        seg1 = label(labels <= 0)
        labels = label2rgb(seg1, image=thresh_vi, bg_label=0) * 2.5
        sigma = 0.
        lb = rescale_intensity(labels,
                               in_range=(-sigma, 1 + sigma),
                               out_range=(0, 1))
        mask = np.where(lb <= 0.3, 0., 1)
        dil_erod_mask = dilate_erode(mask[:, :, 0],
                                     kernel_size=3,
                                     dil_iters=5,
                                     eros_iters=6,
                                     hole_fill=True)
        reduced_mask = reduce_holes(dil_erod_mask * 255) * 255
        return reduced_mask

    def seperate_components(self, mask):
        # Store individual plant components in a list
        mask = mask.astype(np.uint8)
        nb_components, output, _, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        # Remove background component
        nb_components = nb_components - 1
        list_filtered_masks = []
        for i in range(0, nb_components):
            filtered_mask = np.zeros((output.shape))
            filtered_mask[output == i + 1] = 255
            list_filtered_masks.append(filtered_mask)
        return list_filtered_masks

    def component_stats(self, mask, connectivity=2):
        # Creates regionprops table for component
        labels = measure.label(mask, connectivity=connectivity)
        props_dict = measure.regionprops_table(labels, properties=CUTOUT_PROPS)
        return props_dict

    def processing_pipeline(self):
        ##########################################################################
        ## Get images from json files
        for label_set in tqdm(self.labels,
                              desc="Segmenting Vegetation",
                              colour="green",
                              leave=False):

            labelpardir = label_set.parent.parent
            imgpath = Path(f"{labelpardir}/developed/{label_set.stem}.jpg")

            # Get image dataclass with bbox set
            imgdata = ImageData(image_id=imgpath.stem,
                                image_path=imgpath,
                                batch_id=self.batch_id)
            rgb_array = imgdata.array

            ##########################################################################
            ## Process on images by individual bbox detection
            cutout_num = 0
            cutout_ids = []
            bboxes = imgdata.bbox.bboxes
            for box in tqdm(bboxes,
                            leave=False,
                            colour="#6dbc90",
                            desc="Generating Cutouts"):

                y1 = int(box.local_coordinates["top_left"][1])
                y2 = int(box.local_coordinates["bottom_left"][1])
                x1 = int(box.local_coordinates["top_right"][0])
                x2 = int(box.local_coordinates["bottom_right"][0])

                # Crop image to bbox
                rgb_crop = rgb_array[y1:y2, x1:x2]
                #####################################################################
                ## Processing begins
                # Get VI
                v_index = getattr(VegetationIndex(), self.vi)
                vi = v_index(rgb_crop)
                thresh_vi = np.where(vi <= 0, 0, vi)
                thresh_vi = np.where((thresh_vi > 20) & (thresh_vi < 100),
                                     thresh_vi * 2, thresh_vi)
                # Get classified mask
                clalgo = getattr(ClassifyMask(), self.class_algorithm)
                class_mask = clalgo(thresh_vi)
                # Clean edges and reduce holes
                # class_mask = dilate_erode(class_mask,
                #                           kernel_size=3,
                #                           dil_iters=22,
                #                           eros_iters=3,
                #                           hole_fill=False)
                # Clear borders
                if self.clear_border:
                    class_mask = clear_border(class_mask) * 255
                # Create RGB cutout for second round of processing
                cutout_0 = apply_mask(rgb_crop, class_mask, "black")
                ## Second round of processing
                mask = self.reprocess_bbox_cutouts(cutout_0)
                # Separate components
                list_cutouts_masks = self.seperate_components(mask)
                for cut_mask in list_cutouts_masks:
                    new_cutout = apply_mask(cutout_0, cut_mask, "black")
                    new_cropped_cutout = crop_cutouts(new_cutout)
                    cutout_path = self.save_cutout(new_cropped_cutout,
                                                   imgdata.image_path,
                                                   cutout_num)
                    # Save cutout ids for DB
                    cutout_ids.append(cutout_path.stem)
                    # Get cutout stats
                    cutout_dict = self.component_stats(cut_mask)
                    # Reformat regionprops results for json
                    for i in cutout_dict.keys():
                        cutout_dict[i] = str(list(cutout_dict[i])[0])

                    # Create cutout dataclass
                    cutout = Cutout(site_id=self.site_id,
                                    cutout_num=cutout_num,
                                    cutout_path=str(cutout_path.name),
                                    image_id=imgdata.image_id,
                                    days_after_planting=14,
                                    stats=cutout_dict,
                                    date=self.date)
                    if self.db is not None:
                        # Move to database
                        ctdoc = asdict(cutout)
                        self.db.Cutouts.insert_one(ctdoc)
                    cutout_num += 1
            if self.db is not None:
                #Pass images with cutout_ids and paths info to DB
                imgdata.cutout_ids = cutout_ids
                imgdata.image_path = str(imgdata.image_path.name)
                doc = asdict(imgdata)
                self.db.Images.insert_one(doc)


def main(cfg: DictConfig) -> None:
    # Connect to database
    db = getattr(Connect.get_connection(),
                 cfg.general.db) if cfg.general.save_to_database else None

    batch = BatchMetadata(upload_dir=cfg.general.imagedir,
                          site_id=get_site_id(cfg.general.batchdir),
                          upload_datetime=get_upload_datetime(
                              cfg.general.imagedir))
    if db is not None:
        # Insert Batch info into database
        batch_doc = asdict(batch)
        db.Batches.insert_one(batch_doc)

    bcfg = BatchConfigImages(batch, cfg)
    Vegseg = SegmentVegetation(bcfg, db, cfg)
    print("*********************")
