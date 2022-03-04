import os

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config")
def run_SEMIF(cfg: DictConfig) -> None:
    # TODO implement logging
    cfg = OmegaConf.create(cfg)
    # print("Working directory : {}".format(os.getcwd()))
    # print(cfg.general.mask_savedir)
    # Get method from yaml to run module in test_hydrafile.py
    task = get_method(f"{cfg.general.task}.main")
    # Run task
    task(cfg)


if __name__ == "__main__":
    run_SEMIF()
