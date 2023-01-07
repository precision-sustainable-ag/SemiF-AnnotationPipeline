import getpass
import logging
import sys
import traceback

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

sys.path.append("move_data")
sys.path.append("autoSfM")
sys.path.append("segment")
sys.path.append("inspect")

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_PIPELINE(cfg: DictConfig) -> None:

    cfg = OmegaConf.create(cfg)
    whoami = getpass.getuser()
    log.info(f"Starting {cfg.general.task} as {whoami}")
    try:
        # Single task
        tsk = cfg.general.task
        task = get_method(f"{tsk}.main")
        task(cfg)
    except Exception as e:
        log.error(f"Failed: {e} \n{traceback.format_exc()}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    run_PIPELINE()
