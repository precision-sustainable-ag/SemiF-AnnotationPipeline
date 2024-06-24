import glob
import logging
import os
from math import ceil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import piexif
from PIL import Image, ImageFile
from skimage.color import rgb2hsv
from skimage.morphology import binary_closing, square

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
            f"More photos than masks. Removing {len(miss)} extra down_scaled photos"
        )
        for img in miss:
            Path(cfg.asfm.down_photos, img + ".jpg").unlink()
    elif len(add) > len(miss):
        # More masks than photos. Remove extra masks
        log.warning(
            f"More masks than photos. Removing {len(add)} extra down_scaled masks"
        )
        for mask in add:
            Path(cfg.asfm.down_masks, mask + "_mask.png").unlink()


def resize_and_save(data):
    image_src = data["image_src"]
    image_dst = data["image_dst"]
    scale = data["scale"]
    masks = data["masks"]

    assert scale > 0.0 and scale <= 1.0, "scale should be between (0, 1]."

    try:
        image = Image.open(image_src)
        width, height = image.size
        scaled_width, scaled_height = int(ceil(width * scale)), int(
            ceil(height * scale)
        )

        kwargs = {}
        if not masks:
            try:
                exif_data = piexif.load(image.info["exif"])
                exif_bytes = piexif.dump(exif_data)
                kwargs["exif"] = exif_bytes
            except KeyError:
                print("EXIF data not found, resizing without EXIF data.")

            resized_image = image.resize((scaled_width, scaled_height))
            resized_image.save(image_dst, quality=95, **kwargs)
        else:
            resized_image = image.resize((scaled_width, scaled_height))
            resized_image.save(image_dst, **kwargs)
    except (IOError, SyntaxError) as e:
        log.error(f"Bad file: {image_src}")
        return
    except Exception as e:
        log.error(f"Error processing file {image_src}: {e}")
        return


def resize_photo_diretory(cfg):
    base_path = cfg["batchdata"]["images"]
    save_dir = cfg["asfm"]["down_photos"]

    files = glob.glob(os.path.join(base_path, "*.jpg"))
    num_files = len(files)
    log.info(f"Processing {num_files} files.")

    data = [
        {
            "image_src": src,
            "image_dst": os.path.join(save_dir, os.path.basename(src)),
            "scale": cfg["asfm"]["downscale"]["factor"],
            "masks": False,
        }
        for src in files
    ]

    # Adjust the number of processes as needed
    num_processes = int(len(os.sched_getaffinity(0)) / cfg.general.cpu_denominator)

    try:
        with Pool(num_processes) as pool:
            for i, _ in enumerate(pool.imap_unordered(resize_and_save, data), 1):
                # pool.imap_unordered(resize_and_save, data)
                print(f"Progress: {i}/{num_files} images resized")
    except KeyboardInterrupt:
        log.info("Interrupted by user, terminating...")
        pool.terminate()
    except Exception as e:
        log.error(f"An error occurred: {e}")
    finally:
        log.info("Completed resizing images.")


def create_masks(cfg):
    down_photos_dir = cfg["asfm"]["down_photos"]
    save_dir = cfg["asfm"]["down_masks"]
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    img_files = glob.glob(os.path.join(down_photos_dir, "*.jpg"))
    num_files = len(img_files)
    log.info(f"Masking {num_files} images.")
    data = [
        {
            "image_src": src,
            "mask_dst": str(Path(save_dir, Path(src).stem + "_mask.png")),
        }
        for src in img_files
    ]

    # Adjust the number of processes as needed
    num_processes = int(len(os.sched_getaffinity(0)) / cfg.general.cpu_denominator)

    # try:
    with Pool(num_processes) as pool:
        for i, _ in enumerate(pool.imap_unordered(mask_img, data), 1):
            # pool.imap_unordered(resize_and_save, data)
            print(f"Progress: {i}/{num_files} images masked.")
    # except KeyboardInterrupt:
    #     log.info("Interrupted by user, terminating...")
    #     pool.terminate()
    # except Exception as e:
    #     log.error(f"An error occurred: {e}")
    # finally:
    #     log.info("Completed masking images.")


def simple_mask(mask):
    # print(closed_mask)
    # Calculate the number of pixels for 5% of the height of the image
    percent = 0.15
    height_percent = int(mask.shape[0] * percent)
    # Set the top 5% and bottom 5% of the mask to false (unmasked)
    if mask.max() == 255:
        if mask[:height_percent, :].max() == 255:
            mask[:height_percent] = 255

        if mask[-height_percent:, :].max() == 255:
            mask[-height_percent:, :] = 255
    return mask


def mask_img(data):
    image_src = data["image_src"]
    mask_dst = data["mask_dst"]

    image = cv2.cvtColor(cv2.imread(image_src), cv2.COLOR_BGR2RGB)
    # Convert the image to HSV
    hsv_image = rgb2hsv(image)
    # Define the range for blue color
    # These ranges can be adjusted depending on the shade of blue in the image
    lower_blue = np.array([0.4, 0.3, 0.2])
    upper_blue = np.array([0.6, 0.9, 1])

    # Create a binary mask for the blue color
    mask = (
        (hsv_image[:, :, 0] >= lower_blue[0])
        & (hsv_image[:, :, 0] <= upper_blue[0])
        & (hsv_image[:, :, 1] >= lower_blue[1])
        & (hsv_image[:, :, 1] <= upper_blue[1])
        & (hsv_image[:, :, 2] >= lower_blue[2])
        & (hsv_image[:, :, 2] <= upper_blue[2])
    )

    # Convert the mask to uint8 format
    mask = mask.astype(np.uint8)

    kernel = square(35)
    closed_mask = binary_closing(mask, kernel).astype(np.uint8)
    closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.dilate(closed_mask, kernel, iterations=3) * 255
    masked = simple_mask(closed_mask)

    cv2.imwrite(mask_dst, masked)

    return True


def resize_masks(cfg):
    base_path = cfg["batchdata"]["masks"]
    save_dir = cfg["asfm"]["down_masks"]

    files = glob.glob(os.path.join(base_path, "*.png"))
    num_files = len(files)
    log.debug(f"Found {num_files} files to process.")

    data = [
        {
            "image_src": src,
            "image_dst": os.path.join(save_dir, os.path.basename(src)),
            "scale": cfg["asfm"]["downscale"]["factor"],
            "masks": True,
        }
        for src in files
    ]

    # Adjust the number of processes as needed
    num_processes = int(len(os.sched_getaffinity(0)) / cfg.general.cpu_denominator)

    try:
        with Pool(num_processes) as pool:
            # pool.imap_unordered(resize_and_save, data)
            for i, _ in enumerate(pool.imap_unordered(resize_and_save, data), 1):
                print(f"Progress: {i}/{num_files} masks resized")
    except KeyboardInterrupt:
        log.info("Interrupted by user, terminating...")
        pool.terminate()
    except Exception as e:
        log.error(f"An error occurred: {e}")
    finally:
        log.info("Completed resizing masks.")

    remove_missing_data(cfg)
