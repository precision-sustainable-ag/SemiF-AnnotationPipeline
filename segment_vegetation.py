import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dacite import from_dict
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

from semif_utils.datasets import (CUTOUT_PROPS, BatchConfigImages,
                                  BatchMetadata, BBoxMetadata, Cutout,
                                  CutoutProps, ImageData)
from semif_utils.mongo_utils import Connect
from semif_utils.utils import (apply_mask, clear_border, crop_cutouts,
                               dilate_erode, get_site_id, get_upload_datetime,
                               make_exg, make_exg_minus_exr, make_exr,
                               make_kmeans, make_ndi, otsu_thresh,
                               reduce_holes)


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


class GenCutoutProps:

    def __init__(self, mask):
        """ Generate cutout properties and returns them as a dataclass.
        Args:
            mask (array): 2D np array
        
        Returns: 
            cutout_props (object): CutoutProps dataclass with each region prop as attribute
        """
        self.mask = mask

    def from_regprops_table(self, connectivity=2):
        """Generates list of region properties for each cutout mask
        """
        labels = measure.label(self.mask, connectivity=connectivity)
        props = measure.regionprops_table(labels, properties=CUTOUT_PROPS)
        # Get largest region if multiple props detected (rare)
        nprops = {}
        for key, value in props.items():
            if len(value) < 2:
                nprops[key] = float(value)
            else:
                nprops[key] = float(value[0])
        nprops["centroid"] = [nprops["centroid-0"], nprops["centroid-1"]]
        nprops.pop("centroid-0", "centroid-1")
        return nprops

    def to_dataclass(self):
        table = self.from_regprops_table()
        cutout_props = from_dict(data_class=CutoutProps, data=table)
        return cutout_props


class SegmentVegetation:

    def __init__(self, bcfg, db, cfg: DictConfig) -> None:
        """_summary_

        Args:
            bcfg (_type_): _description_
            db (_type_): _description_
            cfg (DictConfig): _description_
        """
        self.bcfg = bcfg
        self.batch_id = bcfg.batch_id
        self.batchdir = Path(cfg.general.batchdir)
        self.imagedir = Path(cfg.general.imagedir)

        self.detectioncsv = self.batchdir / "detections.csv"
        self.labels = [x for x in (self.batchdir / "labels").glob("*.json")]

        self.cutoutdir = self.get_cutoutdir()
        self.clear_border = cfg.segment.clear_border
        self.sitedir, self.site_id = self.get_siteinfo()
        self.vi = bcfg.vi
        self.class_algorithm = bcfg.class_algorithm

        self.db = db
        self.cutout_pipeline()

    def get_cutoutdir(self):
        cutoutdir = Path(self.batchdir, "cutouts")
        cutoutdir.mkdir(parents=True, exist_ok=True)
        return cutoutdir

    def get_siteinfo(self):
        """Uses image directory to gather site specific information.
            Agnostic to what relative path structure is used. As in it does
            not matter whether parent directory of images is sitedir or "developed". 

        Returns:
            sitedir: developed image parent directory name
            site_id: state id takend from sitedir
        """
        states = ['TX', 'NC', 'MD']
        sitedir = [p for st in states for p in self.imagedir.parts
                   if st in p][0]
        site_id = [st for st in states if st in sitedir][0]
        return sitedir, site_id

    def save_cutout(self, cutout, imgpath, cutout_num):
        """Saves cutouts to cutoutdir.

        Args:
            cutout (np.ndarray): final processed cutout array in RGB
            imgpath (str): filename of the original developed image
            cutout_num (int): unique number per image

        Returns:
            cutoutpath: used for saving information to mongodb
        """
        fname = f"{imgpath.stem}_{cutout_num}.png"
        cutout_path = self.cutoutdir / fname
        # return cutout_path
        cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
        return cutout_path

    def seperate_components(self, mask):
        """ Seperates multiple unconnected components in a mask
            for seperate processing. 
        Args:
            mask (np.ndarray):  mask of multiple components form first processing stage

        Returns:
            list[np.ndarray]: list of multiple component masks
        """
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

    def thresh_vi(self, vi, low=20, upper=100, sigma=2):
        """_summary_

        Args:
            vi (np.ndarray): vegetation index single channel
            low (int, optional): lower end of vi threshold. Defaults to 20.
            upper (int, optional): upper end of vi threshold. Defaults to 100.
            sigma (int, optional): multiplication factor applied to range within
                                   "low" and "upper". Defaults to 2.

        Returns:
            thresh_vi: threshold vegetation index
        """
        thresh_vi = np.where(vi <= 0, 0, vi)
        thresh_vi = np.where((thresh_vi > low) & (thresh_vi < upper),
                             thresh_vi * sigma, thresh_vi)
        return thresh_vi

    def process_domain(self, img):
        """First cutout processing step of the full image."""
        ## First Round of processing
        # Get VI
        v_index = getattr(VegetationIndex(), self.vi)
        vi = v_index(img)
        th_vi = self.thresh_vi(vi)
        # Get classified mask
        clalgo = getattr(ClassifyMask(), self.class_algorithm)
        mask = clalgo(th_vi)
        return mask

    def process_cutout(self, cutout):
        """Second cutout processing step.
        """
        # Second round of processing
        vi = make_exg(cutout, thresh=True)
        thresh_vi = np.where(vi <= 0, 0, vi)
        thresh_vi = np.where((thresh_vi > 10) & (thresh_vi < 100),
                             thresh_vi * 5, thresh_vi)
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

    def get_bboxmeta(self, path):
        with open(path) as f:
            j = json.load(f)
            bbox_meta = from_dict(data_class=BBoxMetadata, data=j)
        return bbox_meta

    def cutout_pipeline(self):
        """ Main Processing pipeline. Reads images from list of labels in
            labeldir,             
        """
        for label_set in tqdm(self.labels,
                              desc="Segmenting Vegetation",
                              colour="green"):
            # Get image using label stem and image directory
            imgpath = Path(f"{self.imagedir}/{label_set.stem}.jpg")
            # Get image dataclass with bbox set
            imgdata = ImageData(image_id=imgpath.stem,
                                image_path=imgpath,
                                batch_id=self.batch_id,
                                bbox_meta=self.get_bboxmeta(label_set))

            dt = datetime.strptime(imgdata.exif_meta.DateTime,
                                   "%Y:%m:%d %H:%M:%S")
            # Call image array
            rgb_array = imgdata.array
            ## Process on images by individual bbox detection
            cutout_num = 0
            cutout_ids = []
            bboxes = imgdata.bbox_meta.bboxes
            for box in tqdm(bboxes,
                            leave=False,
                            colour="#6dbc90",
                            desc="Generating Cutouts"):

                x1, y1 = box.local_coordinates["top_left"]
                x2, y2 = box.local_coordinates["bottom_right"]
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                # Crop image to bbox
                rgb_crop = rgb_array[y1:y2, x1:x2]
                mask = self.process_domain(rgb_crop)
                # Clear borders
                if self.clear_border:
                    mask = clear_border(mask) * 255
                # Separate components
                list_cutouts_masks = self.seperate_components(mask)
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
                    cutout_path = self.save_cutout(new_cropped_cutout,
                                                   imgdata.image_path,
                                                   cutout_num)
                    # Save cutout ids for DB
                    cutout_ids.append(cutout_path.stem)
                    # Create dataclass
                    cutout = Cutout(site_id=self.site_id,
                                    cutout_num=cutout_num,
                                    cutout_path=str(cutout_path.name),
                                    image_id=imgdata.image_id,
                                    cutout_props=cutprops,
                                    datetime=dt)
                    cutout_num += 1
                    # Move cutout to database
                    if self.db is not None:
                        to_db(self.db, cutout, "Cutouts")

            # Move image to database
            if self.db is not None:
                imgdata.cutout_ids = cutout_ids
                imgdata.image_path = imgdata.image_path.name
                to_db(self.db, imgdata, "Images")


def to_db(db, data, collection):
    # Inserts dictionaries into mongodb
    data_doc = asdict(data)
    getattr(db, collection).insert_one(data_doc)


def main(cfg: DictConfig) -> None:
    # Create batch metadata
    batch = BatchMetadata(upload_dir=cfg.general.imagedir,
                          site_id=get_site_id(cfg.general.batchdir),
                          upload_datetime=get_upload_datetime(
                              cfg.general.imagedir))
    # Connect to database
    if cfg.general.save_to_database:
        db = getattr(Connect.get_connection(), cfg.general.db)
        to_db(db, batch, "Batches")
    else:
        db = None
    # Run pipeline
    bcfg = BatchConfigImages(batch, cfg)
    vegseg = SegmentVegetation(bcfg, db, cfg)
