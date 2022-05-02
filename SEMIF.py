import os

import hydra
from hydra.utils import get_method, get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import localize_plants  # Do not remove
import remap_labels  # Do not remove
import segment_vegetation  # Do not remove


@hydra.main(config_path="conf", config_name="config")
def run_SEMIF(cfg: DictConfig) -> None:
    # TODO implement logging
    cfg = OmegaConf.create(cfg)
    # TODO major path checking and incremental directory allignment so mask and cutout directories have same timestamp
    if cfg.general.multitask:
        for t in cfg.general.multitasks:
            task = get_method(f"{t}.main")
            # Run task
            task(cfg)
    else:
        print(cfg.general.task)
        task = get_method(f"{cfg.general.task}.main")
        task(cfg)


if __name__ == "__main__":
    run_SEMIF()
