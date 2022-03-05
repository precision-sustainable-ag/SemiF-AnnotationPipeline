import multiprocessing
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from ExtractComponents import ExtractIndCutouts
from GenMaskMultiProc import ExGMaskGenerator, ExGRMaskGenerator
from utils import get_imgs, increment_path

# TODO connect GenerateMask and ExtractComponents pipelines
# TODO Create excutables for both
# TODO make excutable from shell args parser


def get_masks(cfg: DictConfig):
    if cfg.general.inc_paths:
        mask_savedir = cfg.general.mask_savedir
        cfg.general.mask_savedir = str(increment_path(mask_savedir,
                                                      mkdir=True))

    vi = cfg.gen_mask.vi
    # TODO Add more variability and feature options
    mask_gen_class = {
        'exg': ExGMaskGenerator,
        'exgr': ExGRMaskGenerator
    }.get(vi)

    input_imagedir = cfg.general.input_imagedir
    images = get_imgs(input_imagedir, as_strs=True)
    data = [(img, cfg) for img in images]

    pool = multiprocessing.Pool(6)

    tbar = list(
        tqdm(
            pool.imap(mask_gen_class().start_pipeline, data),
            total=len(images),
            desc="Vegetation Mask Generator",
        ))


def get_cutouts(cfg: DictConfig):
    # TODO create checks for same number of images/masks in each dir
    mask_paths = sorted(get_imgs(cfg.general.mask_savedir))
    img_paths = sorted(get_imgs(cfg.general.input_imagedir))
    if cfg.general.inc_paths:
        cutout_dir = cfg.general.cutout_savedir
        cfg.general.cutout_savedir = str(increment_path(cutout_dir,
                                                        mkdir=True))

    for imgp in tqdm(img_paths, desc="Cutout Generator"):
        ExtractIndCutouts(cfg, imgp).extract_and_save_cutout()


def main(cfg: DictConfig) -> None:
    get_masks(cfg)
    get_cutouts(cfg)
