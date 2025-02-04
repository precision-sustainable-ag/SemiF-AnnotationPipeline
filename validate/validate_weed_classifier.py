import logging
import hydra
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Define directories
    data_dir = Path(cfg.data.batchdir)
    images = list((Path(data_dir, "images").glob("*.jpg")))
    plant_detections = list(Path(data_dir, "plant-detections", "processed").glob("*.csv"))

    for plant_detection in plant_detections:
        df = pd.read_csv(plant_detection)
        if df.empty:
            log.info(f"{plant_detection} is empty")
            continue

        unique_classnames = list(df["classifier_classname"].unique())
        if "non_target_weed" in unique_classnames:
            image_path = Path(data_dir, "images", plant_detection.stem + ".jpg")
            image = cv2.imread(str(image_path))
            if image is None:
                log.warning(f"Could not read image: {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            # Create directories for saving outputs
            output_dir = Path(data_dir, "plant-detections", "processed", "full_images_with_insets")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Set up the plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])

            for i, row in df.iterrows():
                if row["classifier_classname"] == "non_target_weed" and row["classifier_confidence"] > 0.99:
                    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                    xmin = int(xmin * w)
                    ymin = int(ymin * h)
                    xmax = int(xmax * w)
                    ymax = int(ymax * h)

                    # Draw a red bounding box on the full image
                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                             linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

                    # Extract the cropout
                    weed_crop = image[ymin:ymax, xmin:xmax]

                    # Add the cropout as an inset
                    imagebox = OffsetImage(weed_crop, zoom=0.3, cmap='gray')
                    ab = AnnotationBbox(imagebox, (xmin, ymin), frameon=True, pad=0.1, bboxprops=dict(edgecolor="black"))
                    ax.add_artist(ab)

            # Save the full image with insets
            full_image_path = output_dir / f"{plant_detection.stem}_inset.jpg"
            plt.savefig(full_image_path, bbox_inches='tight', dpi=300)
            plt.close()

            log.info(f"Processed {plant_detection}, saved full image with insets to {full_image_path}")

if __name__ == "__main__":
    main()
