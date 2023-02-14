import logging
import signal
import sys
import traceback

from omegaconf import DictConfig

from autoSfM.auto_sfm.config_utils import autosfm_present, create_config
from autoSfM.auto_sfm.metashape_utils import SfM
from autoSfM.auto_sfm.resize import resize_masks, resize_photo_diretory

# Set the logger
log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:

    def sigint_handler(signum, frame):
        print("\nPython SIGINT detected. Exiting.\n")
        exit(1)

    def sigterm_handler(signum, frame):
        print("\nPython SIGTERM detected. Exiting.\n")
        exit(1)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Setup config
    cfg = create_config(cfg)

    # Check if autosfm has already been run
    try:
        log.info(f"Checking for autosfm contents")
        if autosfm_present(cfg):
            log.info(
                f"Autosfm has already been run. All contents are available. Moving to next process."
            )
            exit(0)

    except Exception as e:
        log.exception(f"Failed to check asfm contents. Exiting")
        exit(1)

    try:
        if cfg["asfm"]["downscale"]["enabled"]:
            log.info(f"Resizing images")
            resize_photo_diretory(cfg)

            if cfg["asfm"]["use_masking"]:
                log.info(f"Resizing masks")
                resize_masks(cfg)

    except Exception as e:
        log.exception(f"Failed to downsize images. Exiting.")
        exit(1)

    log.info(f"Initializing SfM")
    pipeline = SfM(cfg)

    try:
        log.info(f"Adding photos")
        pipeline.add_photos()
        if cfg["asfm"]["use_masking"]:
            pipeline.add_masks()
    except Exception as e:
        log.exception(f"Failed to add photos. Exiting.")
        exit(1)

    try:
        log.info(f"Detecting markers")
        pipeline.detect_markers()
    except Exception as e:
        log.exception(f"Failed to detect markers. Exiting")
        exit(1)

    try:
        log.info(f"Importing references")
        pipeline.import_reference()
    except Exception as e:
        log.exception(f"Failed to import reference. Exiting")
        exit(1)

    try:
        log.info(f"Aligning photos")
        pipeline.align_photos(correct=True)
    except Exception as e:
        log.exception(f"Failed to align photos. Exiting")
        exit(1)

    try:
        log.info(f"Optimizing cameras")
        pipeline.optimize_cameras()
    except Exception as e:
        log.exception(f"Failed to optimize cameras. Exiting")
        exit(1)

    try:
        log.info(f"Exporting GCP reference")
        pipeline.export_gcp_reference()
    except Exception as e:
        log.exception(f"Failed to export GCP reference. Exiting")
        exit(1)

    try:
        log.info(f"Exporting camera reference")
        pipeline.export_camera_reference()
    except Exception as e:
        log.exception(f"Failed to export camera reference. Exiting")
        exit(1)

    try:
        log.info(f"Exporting error stats")
        pipeline.export_stats()
    except Exception as e:
        log.exception(f"Failed to export error stats. Exiting")
        exit(1)

    # Electives
    try:
        if cfg["asfm"]["depth_map"]["enabled"]:
            log.info(f"Building depth maps")
            pipeline.build_depth_map()
    except Exception as e:
        log.exception(f"Failed to build depth maps. Exiting")
        exit(1)

    try:
        if cfg["asfm"]["dense_cloud"]["enabled"]:
            log.info(f"Buidling dense point cloud")
            pipeline.build_dense_cloud()
    except Exception as e:
        log.exception(f"Failed to buidl dense point cloud. Exiting")
        exit(1)

    try:
        if cfg["asfm"]["dem"]["enabled"]:
            log.info(f"Building DEM")
            pipeline.build_dem()
    except Exception as e:
        log.exception(f"Failed to build DEM. Exiting")
        exit(1)

    try:
        if cfg["asfm"]["orthomosaic"]["enabled"]:
            log.info(f"Building orthomosaic")
            pipeline.build_ortomosaic()
    except Exception as e:
        log.exception(f"Failed to build orthomosaic. Exiting")
        exit(1)

    try:
        if cfg["asfm"]["camera_fov"]["enabled"]:
            log.info(f"Exporting camera FOV information")
            pipeline.camera_fov()
    except Exception as e:
        log.exception(f"Failed to export camera FOV information. Exiting")
        exit(1)

    try:
        log.info(f"Exporting preview image")
        pipeline.capture_view()
    except Exception as e:
        log.exception(f"Failed to export preview image. Exiting")
        exit(1)
    log.info(f"AutoSfM Complete")