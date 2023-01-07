import logging
import os
from pathlib import Path

from omegaconf import DictConfig

log = logging.getLogger(__name__)


class UploadData:

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.batchid = cfg.general.batch_id
        self.upload_script = cfg.movedata.upload.script
        self.ignore_miss = cfg.movedata.ignore_missing
        # Items to upload
        self.miss_results = []
        # Upload paths
        self.asfm_src = cfg.movedata.upload.autosfm
        self.meta_src = cfg.movedata.upload.metadata
        self.mmask_src = cfg.movedata.upload.metamasks
        self.json_src = cfg.movedata.upload.metajson

    def check_autosfm_upload(self):
        """Checks for presences of directories and files
            in batch autosfm directory. Appends missing items
            to self.miss_results.
        """
        necessary_asfm = [
            self.cfg.batchdata.autosfm, self.cfg.asfm.proj_dir,
            self.cfg.asfm.proj_path, self.cfg.asfm.down_photos,
            self.cfg.asfm.down_masks, self.cfg.asfm.refs,
            self.cfg.asfm.gcp_ref, self.cfg.asfm.cam_ref,
            self.cfg.asfm.err_ref, self.cfg.asfm.fov_ref,
            self.cfg.asfm.orthodir, self.cfg.asfm.ortho_path,
            self.cfg.asfm.demdir, self.cfg.asfm.dem_path, self.cfg.asfm.preview
        ]

        for nec in necessary_asfm:
            nec = Path(nec)
            if nec.is_dir():
                is_empty = len(os.listdir(nec)) == 0
                if is_empty:
                    self.miss_results.append(nec)
            elif nec.is_file():
                if not nec.exists():
                    self.miss_results.append(nec)

    def check_data(self):
        """Check existance of autosfm, metadata, meta_masks 
        directories and batch metadata (.json) files. Also checks
        if each is empty. Added to a list (self.miss_local) 
        if they don't exist or are empty.
        """
        # Check batch data
        necessary = [
            self.meta_src, self.mmask_src,
            Path(self.mmask_src, "instance_masks"),
            Path(self.mmask_src, "semantic_masks"), self.json_src
        ]

        for nec in necessary:
            nec = Path(nec)
            if nec.is_dir():
                is_empty = len(os.listdir(nec)) == 0
                if is_empty:
                    self.miss_results.append(nec)
            elif nec.is_file():
                if not nec.exists():
                    self.miss_results.append(nec)
            else:
                self.miss_results.append(nec)

        # Check autosfm data
        self.check_autosfm_upload()

    def upload_autosfm(self):
        os.system(f'bash {self.upload_script} {self.asfm_src} {self.batchid}')

    def upload_metadata(self):
        os.system(f'bash {self.upload_script} {self.meta_src} {self.batchid}')

    def upload_meta_masks(self):
        os.system(f'bash {self.upload_script} {self.mmask_src} {self.batchid}')

    def upload_metajson(self):
        meta_json = Path(self.batchid, self.batchid + ".json")
        os.system(f'bash {self.upload_script} {self.json_src} {meta_json}')
