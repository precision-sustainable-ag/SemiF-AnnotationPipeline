from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from semif_utils import (Relabel_Bbox_Extract_Mask_Colors, apply_mask,
                         clean_mask, clear_border, crop_cutouts,
                         increment_path, make_exg, make_exg_minus_exr,
                         make_exr, make_kmeans, make_ndi, otsu_thresh,
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
        kernel = np.ones(kernel, np.uint8)
        dilmask = cv2.dilate(vi_mask, kernel, iterations=3)
        emask = cv2.erode(dilmask, kernel, iterations=2)
        return emask

    def kmeans(self, vi):
        vi_mask = make_kmeans(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255)
        cleaned_mask = clean_mask(reduce_holes_mask)
        return cleaned_mask


class SegmentVegetation:

    def __init__(self, cfg: DictConfig) -> None:

        datadir = cfg.general.datadir
        workdir = cfg.general.workdir
        projectdir = cfg.general.name
        num_cls = cfg.general.num_classes
        vi_idx = cfg.general.vi
        clsalgo = cfg.general.class_algorithm

        self.num_class = num_cls
        self.vi_name = vi_idx
        self.class_algo = clsalgo
        self.savedir = projectdir
        self.save_maskdir = Path(self.savedir, "cutout_masks")
        self.save_cutoutdir = Path(self.savedir, "cutouts")
        self.img = data[0]
        self.lbls = data[1]
        self.imgpath = data[2]

        self.colors = self.get_colorsmap()
        self.mask = self.processing_pipeline()

    def get_colorsmap(self):
        """colors for segmentation masks"""
        colors = Relabel_Bbox_Extract_Mask_Colors(
            colortheme="journal_of_medicine")
        colors = [list(colors(x)) for x in range(0, self.num_class)]
        return colors

    def remove_empty_masks(self):
        pass

    def save_cutout(self, img_crop, mask, cutout_num):
        cutout = apply_mask(img_crop, mask, "black")
        cutout = crop_cutouts(cutout)
        cutout_fname = f"{self.imgpath.stem}_0{cutout_num}.png"
        cutout_output_path = Path(self.save_cutoutdir, cutout_fname)
        cv2.imwrite(str(cutout_output_path), cutout)

    def reshape_mask(self, mask, empty_mask, lbl):
        species, x1, y1, x2, y2, _ = (int(lbl[0]), int(lbl[1]), int(lbl[2]),
                                      int(lbl[3]), int(lbl[4]), lbl[5])
        mask = np.stack([mask, mask, mask])
        mask = np.moveaxis(mask, 0, -1)
        # rgb_copy[y1:y2, x1:x2][mask[:,:,0] == 255] = colors[species] # For display purposes
        empty_mask[y1:y2, x1:x2][mask[:, :, 0] == 255] = self.colors[species]

    def save_mask(self, mask):
        mask_output_path = Path(self.save_maskdir, self.imgpath.stem + ".png")
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(mask_output_path), mask)

    def processing_pipeline(self):
        rgb_copy = self.img.copy()  # why create a second???
        # Create empty mask
        empty_mask = np.zeros(self.img.shape, dtype=np.uint8)
        cutout_num = 0
        for lbl in self.lbls:
            _, x1, y1, x2, y2, _ = (int(lbl[0]), int(lbl[1]), int(lbl[2]),
                                    int(lbl[3]), int(lbl[4]), lbl[5])
            # Crop image to bbox
            img_sub = rgb_copy[y1:y2, x1:x2]

            # Get VI
            v_index = getattr(VegetationIndex(), self.vi_name)
            vi = v_index(img_sub)

            # Get classified mask
            clalgo = getattr(ClassifyMask(), self.class_algo)
            class_mask = clalgo(vi)

            # Clear borders
            clear_mask = clear_border(class_mask) * 255

            # Remove empty masks
            # TODO create function to remove empty masks and very small mask results

            # Create and save cutout
            self.save_cutout(img_sub, clear_mask, cutout_num)

            # Reshape mask for mapping
            self.reshape_mask(clear_mask, empty_mask, lbl)

            cutout_num += 1

        # Save mask
        self.save_mask(empty_mask)
        return empty_mask


def create_projectdir(workdir, projectdir):
    outputdir = Path(workdir, "runs")
    projectdir = Path(outputdir, projectdir)
    if not outputdir.exists():
        outputdir.mkdir(parents=True)
    if not projectdir.exists():
        projectdir.mkdir(parents=True)
        Path(projectdir, "cutout_masks").mkdir(parents=True)
        Path(projectdir, "cutouts").mkdir(parents=True)
    else:
        projectdir = increment_path(projectdir, sep="_", mkdir=True)

        if not Path(projectdir, "cutout_masks").exists():
            Path(projectdir, "cutout_masks").mkdir(parents=True,
                                                   exist_ok=False)
        if not Path(projectdir, "cutouts").exists():
            Path(projectdir, "cutouts").mkdir(parents=True, exist_ok=False)

    return projectdir


class VegDataIterator:

    def __init__(self, datadir):
        "Image and label iterator for segmenting vegetation from bounding box detection results."
        "Returns images and labels"
        self.datadir = self.check_datadir(Path(datadir))
        self.imgs = get_img_paths(Path(self.datadir, "images"), sort=True)

    def check_datadir(self, datadir):
        assert datadir.exists(), "Data directory does not exist"
        assert Path(datadir,
                    "labels").exists(), "Label directory does not exist"
        assert Path(datadir,
                    "images").exists(), "Image directory does not exist"
        return datadir

    def label_func(self, fname):
        name = fname.stem + ".txt"
        return fname.parent.parent / f"labels/{name}"

    def __getitem__(self, idx):
        # for imgfname in self.imgs:
        imgpath = Path(self.imgs[idx])
        # Load data
        img = cv2.imread(str(self.imgs[idx]))
        # Get image shape info
        img_h, img_w = img.shape[0], img.shape[1]
        #Get bbox info for each image detection result
        lblfname = self.label_func(imgpath)
        lbls = get_bbox(lblfname, img_h, img_w, contains_conf=True)
        return img, lbls, imgpath
