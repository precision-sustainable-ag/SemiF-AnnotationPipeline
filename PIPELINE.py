import getpass
import logging
import sys
import traceback

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from utils.utils import remove_batch, write_batch

sys.path.append("move_data")
sys.path.append("autoSfM")
sys.path.append("segment")
sys.path.append("inspect")

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_PIPELINE(cfg: DictConfig) -> None:

    cfg = OmegaConf.create(cfg)
    whoami = getpass.getuser()
    # read batch from file
    # batches = cfg.batches

    tasks = [k for k, v in cfg.pipeline.items() if v]
    for tsk in tasks:
        log.info(f"Starting {cfg.general.get(tsk)} as {whoami}")
        try:
            task = get_method(f"{tsk}.main")
            task(cfg)
            # Switch find_unprocessed to false so it only runs once

        except Exception as e:
            log.exception("Failed")
            sys.exit(1)
    # write_batch(cfg, cfg.general.batch_id)
    # remove_batch(cfg, cfg.general.batch_id)


if __name__ == "__main__":
    run_PIPELINE()