import logging
import time

from omegaconf import DictConfig

from move_data.utils.download_utils import DownloadData
from move_data.utils.list_batches import ListBatches

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    lb = ListBatches(cfg)
    lb.read_keys()
    # lb.az_list()
    lb.organize_temp()
