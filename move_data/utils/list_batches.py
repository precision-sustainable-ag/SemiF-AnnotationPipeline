import logging
import os
from pathlib import Path
import yaml
from omegaconf import DictConfig
from move_data.utils.data import PipelineKeys

log = logging.getLogger(__name__)


class ListBatches:

    def __init__(self, cfg):
        self.cfg = cfg
        self.keypath = cfg.pipeline_keys
        self.pkeys = self.read_keys()
        self.temp_path = Path("./temp_file.txt")

    def read_keys(self):
        with open(self.keypath, 'r') as file:
            pipe_keys = yaml.safe_load(file)
            sas = pipe_keys['SAS']
            up_cut = sas['cutouts']['upload']
            down_cut = sas['cutouts']['download']

            up_dev = sas['developed']['upload']
            down_dev = sas['developed']['download']
            keys = PipelineKeys(down_dev=down_dev,
                                up_dev=up_dev,
                                down_cut=down_cut,
                                up_cut=up_cut,
                                ms_lic=pipe_keys['metashape']['lic'])
        return keys

    def az_list(self, n=2):
        self.temp_path.touch(exist_ok=True)
        down_dev = self.pkeys.down_dev

        os.system(f"azcopy ls " + f'"{down_dev}"' +
                  f" | cut -d/ -f 1-{n} | awk '!a[$0]++' >> {self.temp_path}")

    def organize_temp(self):
        with open(self.temp_path, 'r') as f:
            lines = [line.rstrip() for line in f]
            lines = [x.replace("INFO: ", "") for x in lines]
            lines = [x.split(";")[0] for x in lines]

        with open(self.temp_path, 'w') as f:
            for line in lines:
                f.write(f"{line}\n")

    def find_unique_batches(Self):
        """"""
