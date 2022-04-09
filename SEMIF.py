import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

import remap_labels  # Do not remove


@hydra.main(config_path="conf", config_name="config")
def run_SEMIF(cfg: DictConfig) -> None:
    # TODO implement logging
    cfg = OmegaConf.create(cfg)
    # TODO major path checking and incremental directory allignment so mask and cutout directories have same timestamp
    task = get_method(f"{cfg.general.task}.main")
    # Run task
    task(cfg)


if __name__ == "__main__":
    run_SEMIF()
