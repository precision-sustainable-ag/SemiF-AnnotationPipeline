import shutil
import os
import subprocess
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf


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

    gcp_src = Path(cfg.data.uploaddir, cfg.general.batch_id, "GroundControlPoints.csv")
    shutil.copy(gcp_src, autosfm_storage)

    # Compose the command
    exe_command = f"docker run \
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
