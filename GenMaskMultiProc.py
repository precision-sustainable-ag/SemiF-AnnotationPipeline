import multiprocessing
from pathlib import Path

import cv2
from omegaconf import DictConfig

from utils import (check_kmeans, exg_minus_exr, get_imgs, make_exg,
                   make_kmeans, make_otsu, read_img, reduce_holes)


class MaskGenerator(object):
    DISPLAY_NAME = "invalid"

    def __init__(self):
        self.meta = None

    def read_image(self, imgpath):
        self.input_img = read_img(imgpath)

    def write_image(self, save_path, img):
        cv2.imwrite(save_path, img)

    def process(self):
        pass  # override in derived classes to perform an actual segmentation

    def start_pipeline(self, args):
        self.img_path, self.mask_savedir, self.viproc, self.clsproc = args  # TODO make checks for paths
        self.read_image(self.img_path)
        # print(
        #     f"filename: {Path(self.img_path).name}\npath: {Path(self.img_path).parent}"
        # )
        return self.process()


class ExGMaskGenerator(MaskGenerator):
    DISPLAY_NAME = "ExG"

    def process(self):
        self.vi = make_exg(self.input_img)

        if "OTSU" in str.upper(self.clsproc):
            self.mask = make_otsu(self.vi)

        elif "KMEANS" in str.upper(self.clsproc):
            self.mask = check_kmeans(make_kmeans(self.vi)) * 255

        self.cleaned_mask = reduce_holes(self.mask)

        mask_savepath = str(Path(self.mask_savedir, Path(self.img_path).name))

        self.write_image(mask_savepath, self.cleaned_mask)
        print(f"ExG saved to : {mask_savepath}\n")


class ExGRMaskGenerator(MaskGenerator):
    DISPLAY_NAME = 'ExGmR'

    def process(self):

        self.vi = exg_minus_exr(self.input_img)

        if "OTSU" in str.upper(self.clsproc):
            self.mask = make_otsu(self.vi)

        elif "KMEANS" in str.upper(self.clsproc):
            self.mask = check_kmeans(make_kmeans(self.vi)) * 255

        self.cleaned_mask = reduce_holes(self.mask)

        mask_savepath = str(Path(self.mask_savedir, Path(self.img_path).name))
        self.write_image(mask_savepath, self.cleaned_mask)
        print(f"ExGmR saved to : {mask_savepath}\n")


# def main(cfg: DictConfig):
#     mask_savedir = Path(cfg.general.mask_savedir)  # save_path
#     vi = cfg.gen_mask.vi
#     class_proc = cfg.gen_mask.classify
#     input_imagedir = cfg.general.input_imagedir

#     mask_gen_class = {
#         'exg': ExGMaskGenerator,
#         'exgr': ExGRMaskGenerator
#     }.get(vi)

#     images = get_imgs(input_imagedir)
#     images = [str(x) for x in images]

#     data = [(img, mask_savedir, vi, class_proc) for img in images]
#     pool = multiprocessing.Pool(6)
#     results = pool.map(mask_gen_class().start_pipeline, data)

#     print('##########FINISHED########')
