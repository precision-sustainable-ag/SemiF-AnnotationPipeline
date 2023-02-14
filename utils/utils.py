from pathlib import Path

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