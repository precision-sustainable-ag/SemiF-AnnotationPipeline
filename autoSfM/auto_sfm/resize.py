import glob
import logging
import os
from math import ceil
from multiprocessing import Pool
from pathlib import Path

import piexif
from PIL import Image

log = logging.getLogger(__name__)


def remove_missing_data(cfg):
    """Removes missing downscaled images are masks.

    Args:
        cfg (DictOmega): Hydra object
    """

    imgs = glob.glob(os.path.join(cfg.asfm.down_photos, "*.jpg"))
    masks = glob.glob(os.path.join(cfg.asfm.down_masks, "*.png"))

    imgs = [Path(img).stem for img in imgs]
    masks = [Path(mask).stem.replace("_mask", "") for mask in masks]

    # find the missing and additional elements in masks
    miss = list(set(imgs).difference(masks))
    add = list(set(masks).difference(imgs))

    if (len(miss) == 0) and (len(add) == 0):
        None
    elif len(miss) > len(add):
        # More photos than masks. Remove extra photos
        log.warning(
            f'More photos than masks. Removing {len(miss)} extra down_scaled photos'
        )
        for img in miss:
            Path(cfg.asfm.down_photos, img + ".jpg").unlink()
    elif len(add) > len(miss):
        # More masks than photos. Remove extra masks
        log.warning(
            f'More masks than photos. Removing {len(add)} extra down_scaled masks'
        )
        for mask in add:
            Path(cfg.asfm.down_masks, mask + "_mask.png").unlink()


def resize_and_save(data):
    image_src = data["image_src"]
    image_dst = data["image_dst"]
    scale = data["scale"]

    assert scale > 0. and scale <= 1., "scale should be between (0, 1]."
    try:
        image = Image.open(image_src)
        width, height = image.size
        scaled_width, scaled_height = int(ceil(width * scale)), int(
            ceil(height * scale))
    except (IOError, SyntaxError) as e:
        log.error('Bad file:',
                  image_src)  # print out the names of corrupt files

    kwargs = {}
    try:
        exif_data = piexif.load(image.info["exif"])
        exif_bytes = piexif.dump(exif_data)
        kwargs["exif"] = exif_bytes
    except KeyError as e:
        print("EXIF data not found, resizing without EXIF data.")

    resized_image = image.resize((scaled_width, scaled_height))
    resized_image.save(image_dst, **kwargs)


def resize_photo_diretory(cfg):
    base_path = cfg["batchdata"]["images"]
    save_dir = cfg["asfm"]["down_photos"]

    files = glob.glob(os.path.join(base_path, "*.jpg"))
    num_files = len(files)
    pool = Pool()
    data = []

    for i, src in enumerate(files):

        log.info(f"Processing {i+1}/{num_files} files.")

        filename = src.split(os.path.sep)[-1]
        dst = os.path.join(save_dir, filename)
        rec = {
            'image_src': src,
            'image_dst': dst,
            'scale': cfg["asfm"]["downscale"]["factor"]
        }
        data.append(rec)
    pool.map(resize_and_save, data)
    pool.close()
    pool.join()


def resize_masks(cfg):

    base_path = cfg["batchdata"]["masks"]
    save_dir = cfg["asfm"]["down_masks"]

    files = glob.glob(os.path.join(base_path, "*.png"))
    num_files = len(files)
    pool = Pool()
    data = []

    for i, src in enumerate(files):

        print(f"Processing {i+1}/{num_files} files.")

        filename = src.split(os.path.sep)[-1]
        dst = os.path.join(save_dir, filename)
        rec = {
            'image_src': src,
            'image_dst': dst,
            'scale': cfg["asfm"]["downscale"]["factor"]
        }
        data.append(rec)
    pool.map(resize_and_save, data)
    pool.close()
    pool.join()

    remove_missing_data(cfg)
