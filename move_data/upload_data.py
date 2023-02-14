import logging
import time
import traceback
from pathlib import Path

from omegaconf import DictConfig

from move_data.utils.upload_utils import UploadData

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:

    start = time.time()
    batch_id = cfg.general.batch_id

    ud = UploadData(cfg)

    log.info(f"Checking data for batch {batch_id}")

    # Check that data exists and is complete

    if not cfg.movedata.ignore_missing:
        try:
            log.info("Checking data prior to upload.")
            ud.check_data()
            if len(ud.miss_results) > 0:

                log.error(
                    f"Upload data incomplete.\nMissing from {batch_id}: {[Path(x).name for x in ud.miss_results]}.\nExiting."
                )
                exit(1)
        except Exception as e:
            log.error(f"Data check failed. Exiting. \n{e}")
            print(traceback.format_exc())
            exit(1)

    try:
        # Upload autosfm directory
        log.info("Uploading AutoSfM data.")
        ud.upload_autosfm()
    except Exception as e:
        log.error(f"AutoSfM upload failed. Exiting.\n{e}")
        exit(1)

    try:
        # Upload metadata
        log.info("Uploading metadata.")
        ud.upload_metadata()
    except Exception as e:
        log.error(f"Metadata upload failed. Exiting.\n{e}")
        exit(1)

    try:
        # Upload meta_masks
        log.info("Uploading meta_masks.")
        ud.upload_meta_masks()
    except Exception as e:
        log.error(f"Meta_masks upload failed. Exiting.\n{e}")
        exit(1)

    try:
        # Upload batch_metadata json
        log.info("Uploading metadata Json.")
        ud.upload_metajson()
    except Exception as e:
        log.error(f"Metadata Json upload failed. Exiting.\n{e}")
        exit(1)

    # TODO: Upload Cutouts
    # batch_id = cfg.blob_names.cutout
    # item = cfg.general.batch_id
    # src = Path(cfg.data.cutoutdir).parent
    # os.system(f'bash {cfg.movedata.workdir}/UploadBatch.sh {batch_id} {item} {src}')

    end = time.time()
    log.info(f"Upload completed in {end - start} seconds.")