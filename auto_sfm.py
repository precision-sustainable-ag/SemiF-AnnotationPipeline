import os
import shutil
import subprocess
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

def main(cfg: DictConfig) -> None:

    save_file = cfg.autosfm.config_save_path
    autosfm_config = OmegaConf.to_container(cfg.autosfm.autosfm_config,
                                            resolve=True)

    with open(save_file, "w") as f:
        yaml.dump(autosfm_config, f)

    # Copy the developed images to a compatible location
    images_src = Path(cfg.data.batchdir, "images")
    autosfm_storage = Path(cfg.autosfm.autosfm_storage, cfg.general.batch_id)
    images_dst = Path(autosfm_storage, "developed")
    shutil.copytree(images_src, images_dst)
    # Get GCP file based on batch ID
    state_id = cfg.general.batch_id.split("_")[0]
    gcp_src = [x for x in Path(cfg.data.utilsdir, state_id).glob("*.csv")][0]
    # Rename csv file to AutoSfM format
    autosfm_storage_csv = autosfm_storage / "GroundControlPoints.csv"
    shutil.copy(gcp_src, autosfm_storage_csv)

    # Compose the command
    try:
        subprocess.check_output('nvidia-smi')
        log.info('Nvidia GPU detected. \nUsing "docker run --gpus all".')
        command_suffix = "docker run --gpus all "
    except Exception: # this command not being found can raise quite a few different errors depending on the configuration
        log.info('No Nvidia GPU in system. Using "docker run"')
        command_suffix = "docker run "
        
    exe_command = f"{command_suffix}\
    -v {cfg.autosfm.autosfm_volume}:/home/psi_docker/autoSfM/volumes \
    -v {cfg.autosfm.autosfm_storage}:/home/psi_docker/autoSfM/storage \
    -v {cfg.autosfm.autosfm_exports}:/home/psi_docker/autoSfM/exports \
    sfm {cfg.autosfm.metashape_key}"

    try:
        # Run the autoSfM command
        subprocess.run(exe_command, shell=True, check=True)
    except Exception as e:
        raise e

    # Make a copy of the autoSfM config for documentation
    copy_file = cfg.autosfm.config_copy_path
    shutil.copy(save_file, copy_file)

    # Copy the exports to the desired location
    export_src = Path(cfg.autosfm.autosfm_exports, cfg.general.batch_id)
    export_dst = Path(cfg.data.batchdir, "autosfm")
    shutil.move(export_src, export_dst)

    # Remove the temp storage
    shutil.rmtree(autosfm_storage)
