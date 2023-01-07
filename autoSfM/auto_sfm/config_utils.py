import os
from pathlib import Path

import yaml
from omegaconf import DictConfig
from utils.utils import read_keys


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_autosfm_dirs(cfg):
    make_dir(cfg.batchdata.autosfm)
    make_dir(cfg.asfm.down_photos)
    make_dir(cfg.asfm.down_masks)
    make_dir(cfg.asfm.proj_dir)
    make_dir(cfg.asfm.refs)
    make_dir(cfg.asfm.orthodir)
    make_dir(cfg.asfm.demdir)


def config_gcp_path(cfg):
    batch_id = cfg["general"]["batch_id"]
    gcp_dir = cfg["data"]["gcp_dir"]
    state_id = batch_id.split("_")[0]
    if state_id == "NC":
        gcp_reference_path = os.path.join(
            gcp_dir, f"GroundControlPoints_NC_2022-07-14_elongated.csv")
    elif state_id == "MD":
        gcp_reference_path = os.path.join(
            gcp_dir, f"GroundControlPoints_MD_2022-06-21_elongated.csv")
    cfg["data"]["gcp_ref"] = gcp_reference_path
    return cfg


def parse_yml(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_config(cfg):
    keys = read_keys(cfg.pipeline_keys)
    cfg.general.metashape_key = keys.ms_lic
    # Prep config
    cfg = config_gcp_path(cfg)
    make_autosfm_dirs(cfg)
    return cfg


def autosfm_present(cfg: DictConfig) -> None:
    """Checks batch autosfm directory for data. Checks for presences of directories and files.

    Args:
        cfg (DictConfig): _description_

    Returns:
        data (list): list of dictionaries
    """
    asfm_list = []
    # asfm dir
    asfm = Path(cfg.batchdata.autosfm)
    asfm_rel = "./" + os.path.relpath(cfg.batchdata.autosfm)
    asfm_ex = asfm.exists()
    asfm_dict = {
        "main_dir": "autosfm",
        "item": "asfm",
        "relative_path": asfm_rel,
        "present": asfm_ex
    }
    asfm_list.append(asfm_dict)
    # PSX project directory
    proj_dir = Path(cfg.asfm.proj_dir)
    proj_dir_rel = "./" + os.path.relpath(cfg.asfm.proj_dir)
    proj_dir_ex = proj_dir.exists()
    proj_dir_dict = {
        "main_dir": "autosfm",
        "item": "proj_dir",
        "relative_path": proj_dir_rel,
        "present": proj_dir_ex
    }
    asfm_list.append(proj_dir_dict)
    # Metashape project psx file
    proj_path = Path(cfg.asfm.proj_path)
    proj_path_rel = "./" + os.path.relpath(cfg.asfm.proj_path)
    proj_path_ex = proj_path.exists()
    proj_path_dict = {
        "main_dir": "autosfm",
        "item": "proj_path",
        "relative_path": proj_path_rel,
        "present": proj_path_ex
    }
    asfm_list.append(proj_path_dict)
    # Downscaled photos
    down_photos = Path(cfg.asfm.down_photos)
    down_photos_rel = "./" + os.path.relpath(cfg.asfm.down_photos)
    down_photos_ex = down_photos.exists()
    down_photos_dict = {
        "main_dir": "autosfm",
        "item": "down_photos",
        "relative_path": down_photos_rel,
        "present": down_photos_ex
    }
    asfm_list.append(down_photos_dict)
    # Downscaled masks
    down_masks = Path(cfg.asfm.down_masks)
    down_masks_rel = "./" + os.path.relpath(cfg.asfm.down_masks)
    down_masks_ex = down_masks.exists()
    down_masks_dict = {
        "main_dir": "autosfm",
        "item": "down_masks",
        "relative_path": down_masks_rel,
        "present": down_masks_ex
    }
    asfm_list.append(down_masks_dict)
    # References
    refs = Path(cfg.asfm.refs)
    refs_rel = "./" + os.path.relpath(cfg.asfm.refs)
    refs_ex = refs.exists()
    refs_dict = {
        "main_dir": "autosfm",
        "item": "refs",
        "relative_path": refs_rel,
        "present": refs_ex
    }
    asfm_list.append(refs_dict)
    # GCP ref
    gcp_ref = Path(cfg.asfm.gcp_ref)
    gcp_ref_rel = "./" + os.path.relpath(cfg.asfm.gcp_ref)
    gcp_ref_ex = gcp_ref.exists()
    gcp_ref_dict = {
        "main_dir": "autosfm",
        "item": "gcp_ref",
        "relative_path": gcp_ref_rel,
        "present": gcp_ref_ex
    }
    asfm_list.append(gcp_ref_dict)
    # Camera reference
    cam_ref = Path(cfg.asfm.cam_ref)
    cam_ref_rel = "./" + os.path.relpath(cfg.asfm.cam_ref)
    cam_ref_ex = cam_ref.exists()
    cam_ref_dict = {
        "main_dir": "autosfm",
        "item": "cam_ref",
        "relative_path": cam_ref_rel,
        "present": cam_ref_ex
    }
    asfm_list.append(cam_ref_dict)
    # Error Reference
    err_ref = Path(cfg.asfm.err_ref)
    err_ref_rel = "./" + os.path.relpath(cfg.asfm.err_ref)
    err_ref_ex = err_ref.exists()
    err_ref_dict = {
        "main_dir": "autosfm",
        "item": "err_ref",
        "relative_path": err_ref_rel,
        "present": err_ref_ex
    }
    asfm_list.append(err_ref_dict)
    # FOV reference
    fov_ref = Path(cfg.asfm.fov_ref)
    fov_ref_rel = "./" + os.path.relpath(cfg.asfm.fov_ref)
    fov_ref_ex = fov_ref.exists()
    fov_ref_dict = {
        "main_dir": "autosfm",
        "item": "fov_ref",
        "relative_path": fov_ref_rel,
        "present": fov_ref_ex
    }
    asfm_list.append(fov_ref_dict)
    # Ortho directory
    orthodir = Path(cfg.asfm.orthodir)
    orthodir_rel = "./" + os.path.relpath(cfg.asfm.orthodir)
    orthodir_ex = orthodir.exists()
    orthodir_dict = {
        "main_dir": "autosfm",
        "item": "orthodir",
        "relative_path": orthodir_rel,
        "present": orthodir_ex
    }
    asfm_list.append(orthodir_dict)
    # Ortho tiff
    ortho_path = Path(cfg.asfm.ortho_path)
    ortho_path_rel = "./" + os.path.relpath(cfg.asfm.ortho_path)
    ortho_path_ex = ortho_path.exists()
    ortho_path_dict = {
        "main_dir": "autosfm",
        "item": "ortho_path",
        "relative_path": ortho_path_rel,
        "present": ortho_path_ex
    }
    asfm_list.append(ortho_path_dict)
    # DEM directory
    demdir = Path(cfg.asfm.demdir)
    demdir_rel = "./" + os.path.relpath(cfg.asfm.demdir)
    demdir_ex = demdir.exists()
    demdir_dict = {
        "main_dir": "autosfm",
        "item": "demdir",
        "relative_path": demdir_rel,
        "present": demdir_ex
    }
    asfm_list.append(demdir_dict)
    # DEM tiff
    dem_path = Path(cfg.asfm.dem_path)
    dem_path_rel = "./" + os.path.relpath(cfg.asfm.dem_path)
    dem_path_ex = dem_path.exists()
    dem_path_dict = {
        "main_dir": "autosfm",
        "item": "dem_path",
        "relative_path": dem_path_rel,
        "present": dem_path_ex
    }
    asfm_list.append(dem_path_dict)
    # Preview image
    # preview = Path(cfg.asfm.preview)
    # preview_rel = "./" + os.path.relpath(cfg.asfm.preview)
    # preview_ex = preview.exists()
    # preview_dict = {
    #     "main_dir": "autosfm",
    #     "item": "preview",
    #     "relative_path": preview_rel,
    #     "present": preview_ex
    # }
    # asfm_list.append(preview_dict)
    presence = [x["present"] for x in asfm_list]
    return True if all(presence) else False


def parse_and_generate_config(config):

    data_dir = config["data"]["datadir"]
    batch_id = config["general"]["batch_id"]
    gcp_dir = config["gcp_dir"]
    state_id = batch_id.split("_")[0]

    dev_img_dir = os.path.join(data_dir, "semifield-developed-images")
    project_name = batch_id
    batch_dir = os.path.join(dev_img_dir, batch_id)
    batch_autosfm = config.batch_autosfm

    config["base_path"] = batch_autosfm
    config["project_name"] = project_name

    make_dir(batch_autosfm)

    photo_directory = os.path.join(batch_dir, "images")
    masks_directory = os.path.join(batch_dir, "masks")

    if state_id == "NC":
        gcp_reference_path = os.path.join(
            gcp_dir, f"GroundControlPoints_NC_2022-07-14_elongated.csv")
    elif state_id == "MD":
        gcp_reference_path = os.path.join(
            gcp_dir, f"GroundControlPoints_MD_2022-06-21_elongated.csv")

    config["photo_directory"] = photo_directory
    config["masks_directory"] = masks_directory
    config["gcp_reference_path"] = gcp_reference_path

    if config["use_masking"]:
        assert os.path.exists(masks_directory)

    # Add the required keys
    if not config.get("project_path", ""):
        project_path = os.path.join(batch_autosfm, "project")
        make_dir(project_path)
        config["project_path"] = os.path.join(project_path,
                                              config["project_name"] + ".psx")

    if not config.get("camera_export_path", ""):
        reference_path = os.path.join(batch_autosfm, "reference")
        make_dir(reference_path)
        config["camera_export_path"] = os.path.join(reference_path,
                                                    "camera_reference.csv")

    if not config.get("gcp_export_path", ""):
        reference_path = os.path.join(batch_autosfm, "reference")
        make_dir(reference_path)
        config["gcp_export_path"] = os.path.join(reference_path,
                                                 "gcp_reference.csv")

    if not config.get("error_statistics_path", ""):
        reference_path = os.path.join(batch_autosfm, "reference")
        make_dir(reference_path)
        config["error_statistics_path"] = os.path.join(reference_path,
                                                       "error_statistics.csv")

    if config["dem"]["enabled"] and config['dem']["export"]["enabled"]:
        dem_config = config["dem"]["export"]
        if not dem_config.get("path", ""):
            dem_image_path = os.path.join(batch_autosfm, "dem")
            make_dir(dem_image_path)
            dem_config["path"] = os.path.join(dem_image_path, "dem.tif")
        config['dem']["export"] = dem_config

    if config["orthomosaic"]["enabled"] and config['orthomosaic']["export"][
            "enabled"]:
        orthomosaic_config = config["orthomosaic"]["export"]
        if not orthomosaic_config.get("path", ""):
            image_path = os.path.join(batch_autosfm, "ortho")
            make_dir(image_path)
            orthomosaic_config["path"] = os.path.join(image_path,
                                                      "orthomosaic.tif")
        config['orthomosaic']["export"] = orthomosaic_config

    if config["camera_fov"]["enabled"]:
        if not config["camera_fov"].get("camera_fov_path", ""):
            reference_path = os.path.join(batch_autosfm, "reference")
            make_dir(reference_path)
            config["camera_fov"]["camera_fov_path"] = os.path.join(
                reference_path, "fov.csv")

    if config["downscale"]["enabled"]:
        assert not config["downscale"].get(
            "destination",
            ""), "The downscale path should not be manualy specified."
        downscale_dir = os.path.join(batch_autosfm, "downscaled_photos")
        make_dir(downscale_dir)
        config["downscale"]["destination"] = downscale_dir

        if config["use_masking"]:
            downscaled_mask_dir = os.path.join(batch_autosfm,
                                               "downscaled_masks")
            make_dir(downscaled_mask_dir)
            config["downscale"]["mask_destination"] = downscaled_mask_dir

    config["logfile"] = os.path.join(batch_autosfm, "logfile.log")

    return config