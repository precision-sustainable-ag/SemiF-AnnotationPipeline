from dataclasses import dataclass, field
from multiprocessing import Manager, Pool, Process, cpu_count
from pathlib import Path

import cv2
from omegaconf import DictConfig

from semif_utils.utils import (apply_mask, clear_border, crop_cutouts,
                               dilate_erode, get_image_meta,
                               get_upload_datetime, get_watershed, make_exg,
                               reduce_holes, seperate_components, thresh_vi)


def pipeline(data):

    path = Path(data["img"])
    savepath = f"{data['savedir']}/{path.name}"

    # Pipeline
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    exg_vi = make_exg(img, thresh=True)
    th_vi = thresh_vi(exg_vi)

    cv2.imwrite(savepath, th_vi)


imgs = [
    x for x in Path(
        "data/semifield-developed-images/MD_2022-06-21/images").glob("*.jpg")
]
savedir = Path("data/semifield-cutouts/MD_2022-06-21/test")
savedir.mkdir(parents=True, exist_ok=True)

payloads = []
for idx, img in enumerate(imgs[:3]):
    data = {"img": img, "idx": idx, "savedir": savedir}
    payloads.append(data)

procs = cpu_count()
# for cut_mask in list_cutouts_masks:
print("[INFO] launching pool using {} processes...".format(procs))
pool = Pool(processes=procs)
pool.map(pipeline, payloads)
# close the pool and wait for all processes to finish
print("[INFO] waiting for processes to finish...")
pool.close()
pool.join()
print("[INFO] multiprocessing complete")
