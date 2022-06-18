import os
import shutil
from pathlib import Path

from omegaconf import DictConfig

def main(cfg: DictConfig) -> None:

    batch_ids = cfg.maintenance.batch_ids

    for batch_id in batch_ids:

        # Remove upload
        src = Path(cfg.data.uploaddir, batch_id)
        shutil.rmtree(src)

        # Remove developed
        src = Path(cfg.data.developeddir, batch_id)
        shutil.rmtree(src)

        # Remove cutouts
        src = Path(cfg.data.cutoutdir, batch_id)
        shutil.rmtree(src)
