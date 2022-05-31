import os
import shutil
from pathlib import Path

from omegaconf import DictConfig

def main(cfg: DictConfig) -> None:

    # Copy autosfm
    src = Path(cfg.autosfm.autosfmdir)
    dst = Path(cfg.blob_storage.developeddir, cfg.general.batch_id)
    shutil.copy(src, dst)
    # Copy the mapping metadata
    src = Path(cfg.data.developeddir, cfg.general.batch_id, "metadata")
    dst = Path(cfg.blob_storage.developeddir, cfg.general.batch_id)
    shutil.copy(src, dst)

    # Copy cutouts
    src = Path(cfg.data.developeddir, cfg.general.batch_id, "cutouts")
    dst = Path(cfg.blob_storage.cutoutdir, cfg.general.batch_id)
    for cutout in os.listdir(src):
        cutout_src = Path(src, cutout)
        cutout_dst = Path(dst, cutout)
        shutil.copy(cutout_src, cutout_dst)

    # Copy synthetic images
    # TODO
