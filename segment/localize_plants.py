import logging
import time
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def process_csv(csv_f):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_f)
    # Define columns to check
    bbox_cols = ["xmin", "ymin", "xmax", "ymax"]
    all_present = set(bbox_cols).issubset(df.columns)
    if not all_present:
        return pd.DataFrame()

    # Apply constraints
    for col in bbox_cols:
        df[col] = df[col].clip(lower=0, upper=1)  # Clip values to be within [0, 1]

    return df


def main(cfg: DictConfig) -> None:
    start = time.time()
    ## Define directories
    batchdir = Path(cfg.data.batchdir)
    detectiondir = Path(batchdir, "autosfm")
    csv_savepath = Path(detectiondir, "detections.csv")

    # Use detection results if they already exists
    plant_detdir = Path(cfg.data.batchdir, "plant-detections")
    if plant_detdir.exists() and any(plant_detdir.iterdir()):
        detection_dir = Path(cfg.data.batchdir, "plant-detections")
        detections = [x for x in detection_dir.glob("*.csv")]
        dfs = []
        for det in detections:
            # df = pd.read_csv(det)
            df = process_csv(det)
            columns_names = [
                "bounding_box_id",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "conf",
                "class",
                "classname",
            ]
            present_columns = df.columns
            all_present = set(columns_names).issubset(present_columns)
            if not all_present:
                continue

            df["imgname"] = det.stem + ".jpg"
            df = df.rename(columns={"classname": "name"})
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(csv_savepath)
        log.info(f"Combined detections and saved to: \n{csv_savepath}")
    else:
        log.error("No detections present. Exiting.")
        exit(1)

    end = time.time()
    log.info(f"Localize plants completed in {end - start} seconds.")
