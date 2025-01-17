import logging
import hydra
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from omegaconf import DictConfig


log = logging.getLogger(__name__)


# Define directories and load data
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Define directories
    data_dir = Path(cfg.data.batchdir)
    images = list((Path(data_dir,"images").glob("*.jpg")))
    plant_detections = list(Path(data_dir,"plant-detections", "processed").glob("*.csv"))
    for plant_detection in plant_detections:
        df = pd.read_csv(plant_detection)
        if df.empty:
            log.info(f"{plant_detection} is empty")
            continue
        unique_classnames = list(df["classifier_classname"].unique())
        if "non_target_weed" in unique_classnames:
            image_path = Path(data_dir,"images", plant_detection.stem + ".jpg")
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            weed_crop_dir = Path(data_dir, "plant-detections", "processed", "non_target_weed")
            weed_crop_dir.mkdir(parents=True, exist_ok=True)
            for i, row in df.iterrows():
                if row["classifier_classname"] == "non_target_weed":
                    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                    xmin = int(xmin * w)
                    ymin = int(ymin * h)
                    xmax = int(xmax * w)
                    ymax = int(ymax * h)
                    weed_crop = image[ymin:ymax, xmin:xmax]
                    weed_crop_path = weed_crop_dir / f"{plant_detection.stem}_{i}.jpg"
                    cv2.imwrite(str(weed_crop_path), cv2.cvtColor(weed_crop, cv2.COLOR_RGB2BGR))
            log.info(f"Processed {plant_detection} and saved to {weed_crop_path}")
                    
    
if __name__ == "__main__":
    main()