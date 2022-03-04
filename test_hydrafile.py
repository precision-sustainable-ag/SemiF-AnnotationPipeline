from omegaconf import DictConfig, OmegaConf, open_dict


def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.general.task)
