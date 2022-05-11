import shutil
from omegaconf import DictConfig, OmegaConf
import yaml


def main(cfg: DictConfig) -> None:

    save_file = cfg.autosfm.config_save_path
    autosfm_config = OmegaConf.to_container(cfg.autosfm.autosfm_config, resolve=True)

    with open(save_file, "w") as f:
        yaml.dump(autosfm_config, f)

    # TODO: subprocess to run autoSfM docker container

    # Make a copy of the autoSfM config for documentation
    copy_file = cfg.autosfm.config_copy_path
    shutil.copy(save_file, copy_file)