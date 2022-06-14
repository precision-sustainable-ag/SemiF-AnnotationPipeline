import os
import shutil
from pathlib import Path

from omegaconf import DictConfig

def main(cfg: DictConfig) -> None:

    # Copy autosfm
    src = Path(cfg.autosfm.autosfmdir)
    dst = Path(cfg.blob_storage.developeddir, cfg.general.batch_id, "autosfm")
    shutil.copytree(src, dst)

    # Copy the mapping metadata
    src = Path(cfg.data.developeddir, cfg.general.batch_id, "metadata")
    dst = Path(cfg.blob_storage.developeddir, cfg.general.batch_id, "metadata")
    shutil.copytree(src, dst)

    # Copy json
    src = Path(cfg.data.batchdir, f"{cfg.general.batch_id}.json")
    dst = Path(cfg.blob_storage.developeddir, cfg.general.batch_id)
    shutil.copy(src, dst)

    # Copy cutouts
    src = Path(cfg.data.cutoutdir, cfg.general.batch_id)
    dst = Path(cfg.blob_storage.cutoutdir, cfg.general.batch_id)
    shutil.copytree(src, dst)

    # Copy the shapefiles if needed
    location = cfg.general.batch_id.split("_")[0]
    src = Path(cfg.data.utilsdir, location, "shapefiles")
    dst = Path(Path(cfg.blob_storage.utilsdir, location, "shapefiles"))
    if not dst.exists():
        shutil.copytree(src, dst)

    # TODO: Copy synthetic images
