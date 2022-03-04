import multiprocessing
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mongoengine import DynamicDocument, connect
from omegaconf import DictConfig, OmegaConf

from ExtractComponents import ContourCollection, ExtractCutouts
from GenMaskMultiProc import ExGMaskGenerator, ExGRMaskGenerator
from utils import get_imgs

# TODO connect GenerateMask and ExtractComponents pipelines
# TODO Create excutables for both
# TODO make excutable from shell args parser


def get_masks(cfg: DictConfig):

    mask_savedir = Path(cfg.general.mask_savedir)  # save_path
    vi = cfg.gen_mask.vi
    class_proc = cfg.gen_mask.classify
    input_imagedir = cfg.general.input_imagedir

    # TODO Add more variability and feature options
    mask_gen_class = {
        'exg': ExGMaskGenerator,
        'exgr': ExGRMaskGenerator
    }.get(vi)

    images = get_imgs(input_imagedir)
    images = [str(x) for x in images]

    data = [(img, mask_savedir, vi, class_proc) for img in images]
    pool = multiprocessing.Pool(6)
    results = pool.map(mask_gen_class().start_pipeline, data)
    print('##########FINISHED########')


# # Define where masks are located
# mask_dir = "data/test_results"


def get_cutouts(cfg: DictConfig):
    masks_paths = get_imgs(cfg.general.mask_savedir)
    for mask_path in masks_paths:
        # Extract components
        ecut = ExtractCutouts(mask_path)
        # Assign document fields
        cutout_list_obj = ContourCollection(mask_path)

        cutouts = cutout_list_obj.cutout_list
        print(cutouts)

        # Uncomment to assign/save/update
        # img.cutouts = cutout_list_obj.cutout_list
        # img.update(unset__vegetation_cutouts=True)
        # img.save()


        # Extract contours from database and view
        # for cutout in img.cutouts:
        #     orig_img = plt.imread(f"data/sample/{img.file_name}")
        #     arr_cntr = np.array(cutout["contours"], dtype=np.int32)
        #     cv2.drawContours(orig_img, np.array([arr_cntr], dtype=np.int32),
        #                      -1, (0, 0, 255), 10)
        #     plt.imshow(orig_img)
        #     plt.show()
def main(cfg: DictConfig):
    #     print(OmegaConf.to_yaml(cfg))
    get_masks(cfg)
    get_cutouts(cfg)
