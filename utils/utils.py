from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from utils.data import PipelineKeys


def read_keys(keypath):
    with open(keypath, 'r') as file:
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

def read_yaml(keypath):
    with open(keypath, 'r') as file:
        cfg = yaml.safe_load(file)
        
        temp_path = Path(cfg["find_missing"]["container_list"])
        print(cfg)
        pkeys = cfg.movedata.SAS_keys
        # Required data directories to be considered "processed".
        # Should be sub-directories of batch_id in Azure
        batch_data = cfg.movedata.find_missing.processed_data
    return temp_path, pkeys, batch_data

def remove_batch(cfg, batch):
    with open(cfg.logs.unprocessed, "r") as f:
        lines = f.readlines()
    with open(cfg.logs.unprocessed, "w") as f:
        for line in lines:
            if line.strip("\n") != batch:
                f.write(line)


def write_batch(cfg, batch):
    with open(cfg.logs.processed, 'a') as f:
        f.write(f"{batch}\n")


def cutout_csvs2df(cutout_dir):
    """Globs cutout dir csvs from main cutout dir, creates dataframes
    for each one, then concatenates them all.

    Args:
        cutout_dir (_type_): _description_
    """
    data = Path(cutout_dir).glob("*")
    csvs = []
    for a in data:
        csv = list(a.glob("*.csv"))
        if len(csv) > 0:
            csvs.append(csv[0])
    df = pd.concat([pd.read_csv(x, low_memory=False) for x in csvs])


def trans_cutout(img):
    """ Get transparent cutout from cutout image with black background. Requires RGB image"""

    # img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    # threshold on black to make a mask
    color = (0, 0, 0)
    mask = np.where((img == color).all(axis=2), 0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result