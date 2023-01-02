import logging
import os

import hydra
from hydra.utils import get_method, get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import auto_sfm
import develop_images
import localize_plants  # Do not remove
import remap_labels  # Do not remove
import assign_species
import segment_vegetation  # Do not remove

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_SEMIF(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    if cfg.general.multitask:
        for t in cfg.general.multitasks:
            task = get_method(f"{t}.main")
            # Run task
            task(cfg)
    else:
        print(cfg.general.task)
        log.info(f"Starting task {cfg.general.task}")
        task = get_method(f"{cfg.general.task}.main")
        task(cfg)


if __name__ == "__main__":
    run_SEMIF()
