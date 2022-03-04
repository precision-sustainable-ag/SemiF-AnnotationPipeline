########################################################################
""" Using the config object """
# from omegaconf import DictConfig, OmegaConf, open_dict
# @hydra.main(config_path="config")
# def my_app(cfg: DictConfig) -> None:
#     assert cfg.db.collection == "images"  # attribute style access
#     assert cfg["db"]["collection"] == "images"  # dictionary style access
#     assert isinstance(cfg.db.image_directory, str)  # Value interpolation type
#     # Print all contents of config file
#     print(OmegaConf.to_yaml(cfg))
#     # cfg.db.waldo  # raises an exception

########################################################################
"""Using config groups"""
# from omegaconf import DictConfig, OmegaConf, open_dict

# @hydra.main(config_path="config")
# def my_app(cfg: DictConfig) -> None:
#     # Print all contents of config file
#     print(OmegaConf.to_yaml(cfg))

########################################################################
"""Selecting default configs"""
# @hydra.main(config_path="conf", config_name="config")
# def my_app(cfg: DictConfig) -> None:
#     print(OmegaConf.to_yaml(cfg))

########################################################################
"""To run the same application with multiple different configurations"""
# python my_app_hydra_walkthrough.py -m db=mongodb,mysql schema=clover,sunflower metadata=full, part
## Additional sweep types
# Hydra supports other kinds of sweeps, e.g:
# x=range(1,10)                  # 1-9
# schema=glob(*)                 # warehouse,support,school
# schema=glob(*,exclude=w*)      # support,school
# python my_app_hydra_walkthrough.py -m db=mongodb,mysql schema=glob(*, exclude=s*) metadata=full, part

########################################################################
"""Output/Working directory
Hydra solves the problem of your needing to specify a new output directory for each run, 
by creating a directory for each run and executing your code within that working directory.
https://hydra.cc/docs/configure_hydra/workdir/
"""
# import os

# from hydra.utils import get_original_cwd, to_absolute_path

# @hydra.main(config_path="conf", config_name="config")
# def my_app(cfg: DictConfig) -> None:
#     print(f"Current working directory : {os.getcwd()}")
#     print(f"Orig working directory    : {get_original_cwd()}")
#     print(f"to_absolute_path('foo')   : {to_absolute_path('foo')}")
#     print(f"to_absolute_path('/foo')  : {to_absolute_path('/foo')}")

########################################################################
""" Logging 
By default, Hydra logs at the INFO level to both the console and a log 
file in the automatic working directory
"""
# import logging

# import hydra
# from omegaconf import DictConfig

# # A logger for this file
# log = logging.getLogger(__name__)

# @hydra.main(config_path="conf", config_name="config")
# def my_app(_cfg: DictConfig) -> None:
#     log.info("Info level message")
#     log.debug("Debug level message")

########################################################################
"""Tab completion"""
# eval "$(python my_app_hydra_walkthrough.py -sc install=bash)"

########################################################################
"""Run module from parsing config.yaml file"""
# import hydra
# from hydra.utils import get_method
# from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_path="conf", config_name="config")
# def my_app_hydra_walkthrough(cfg: DictConfig) -> None:
#     # TODO implement logging
#     cfg = OmegaConf.create(cfg)
#     # Get method from yaml to run module in test_hydrafile.py
#     task = get_method(f"{cfg.general.mode}_{cfg.general.task}.main")
#     # Run task
#     task(cfg)

# if __name__ == "__main__":
#     my_app_hydra_walkthrough()
