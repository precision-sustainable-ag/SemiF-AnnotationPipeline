import cv2
import pandas as pd
from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig
import shutil

import logging
log = logging.getLogger(__name__)


class CVATDataGenerator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        # Batch ID.
        self.batch_id = cfg.general.batch_id

        # Full-sized data directories.
        self.image_dir = Path(cfg.batchdata.images)
        self.metadata_dir = Path(cfg.batchdata.metadata)
        self.semantic_mask_dir = Path(cfg.batchdata.meta_masks) / "semantic_masks"

        # Developed-images batch directory.
        self.developed_dir = cfg.data.batchdir
        # Cutout data directory.
        self.cutout_dir = Path(cfg.batchdata.cutouts)

        # Validation results.
        self.validation_results = Path(self.developed_dir) / f"{self.batch_id}_validation_results.csv"

        # Relabel species.
        self.relabel_common_names = ["_".join(name.split()) for name in cfg.cvat.relabel_common_names]

        # Uniqe mask values.
        self.unique_mask_values = set()

        # Filtering options
        self.bbox_size_min_max = cfg.cvat.size_classes
        self.sample_n = cfg.cvat.sample_n

        self.images_size = cfg.cvat.images_size

        self.cutouts = dict()

        # CVat directory name.
        self.cvat_dir_name = self.create_folder_name()
        self.cvat_data_dir = Path(self.cfg.general.workdir) / "cvat" / "data" / self.batch_id / self.cvat_dir_name
        
        # Output directory.
        self.setup_output_directories()

    def create_folder_name(self) -> str:
        """Create a folder name based on config information."""
        common_names = "_".join(self.relabel_common_names)
        sample_size = self.sample_n
        size_class_labels = []

        for label, (min_val, max_val) in self.cfg.cvat.size_class_name_mapping.items():
            if self.bbox_size_min_max.min == min_val and self.bbox_size_min_max.max == max_val:
                size_class_labels.append(label)

        size_class_label = "_".join(size_class_labels)
        folder_name = f"{sample_size}_{size_class_label}_{common_names}_{self.batch_id}_annotations_camvid"
        return folder_name
    
    def setup_output_directories(self) -> None:
        """Setup output directories."""
        self.image_dir_name = "default"
        self.mask_dir_name = "defaultannot"
        self.output_image_dir = Path(self.cvat_data_dir) / self.image_dir_name
        self.output_mask_dir = Path(self.cvat_data_dir) / self.mask_dir_name
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.output_mask_dir.mkdir(parents=True, exist_ok=True)

    def read_validation_results(self) -> pd.DataFrame:
        """Read validation results."""
        return pd.read_csv(self.validation_results)
    
    def read_cutout_data(self) -> pd.DataFrame:
        """Read cutout data."""
        return pd.read_csv(self.cutout_dir / f"{self.batch_id}.csv")

    def get_images_to_relabel(self) -> pd.DataFrame:
        """Get list of images that need to be relabeled."""
        validation_results = self.read_validation_results()
        relabel_images_df = validation_results[validation_results["Status"] == "review"]
        return relabel_images_df

    def filter_by_bbox_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by bbox area."""
        return df[
            (df["cutout_props_bbox_area_cm2"] >= self.bbox_size_min_max.min)
            & (df["cutout_props_bbox_area_cm2"] <= self.bbox_size_min_max.max)
        ]
    def df_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample dataframe."""
        return df.sample(n=self.sample_n).reset_index(drop=True) if df.shape[0] > self.sample_n else df

    def filter_images_by_species(self) -> pd.DataFrame:
        """Filter images by species."""
        relabel_images_df = self.read_cutout_data()
        relabel_images_df["category_common_name"] = relabel_images_df["category_common_name"].str.lower()
        # Remove the _ from the self.relabel_common_names
        relabel_common_names = [name.replace("_", " ") for name in self.relabel_common_names]
        normalized_common_names = [name.lower() for name in relabel_common_names]
        filtered_df = relabel_images_df[relabel_images_df["category_common_name"].isin(normalized_common_names)]

        filtered_df = self.filter_by_bbox_area(filtered_df)
        
        filtered_df = self.df_sample(filtered_df)

        log.info(f"Filtered {filtered_df.shape[0]} images by species.")
        return filtered_df

    def processCutouts(self) -> None:
        """
        Pad or tile crop cutouts to the target size (512x512).
        
        - If the image is larger than target_size in at least one dimension, tile-crop the image:
          For each tile, if it is not full size (e.g. at the image’s borders), pad it on the bottom/right.
        - If the image is smaller than or equal to target_size in both dimensions, pad it (centered) to reach target_size.
        """
        target_size = self.images_size
        cutout_data = self.filter_images_by_species()
        for index, row in cutout_data.iterrows():
            image_path = self.cutout_dir / f"{row['cutout_id']}.jpg"
            image = cv2.imread(str(image_path))
            mask_path = self.cutout_dir / f"{row['cutout_id']}_mask.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Get the unique mask value that isn't 0
            category_common_name = row["category_common_name"]
            unique_mask_values = np.unique(mask)
            for unique_mask_value in unique_mask_values:
                if unique_mask_value != 0:
                    self.unique_mask_values.add((unique_mask_value, category_common_name))

            # Skip if image or mask could not be loaded.
            if image is None or mask is None:
                print(f"Error reading image or mask for cutout_id {row['cutout_id']}")
                continue

            height, width = image.shape[:2]
            
            # If either dimension is larger than target_size, tile-crop the image.
            if height > target_size or width > target_size:
                for i in range(0, height, target_size):
                    for j in range(0, width, target_size):
                        # Use slicing with min() so that if we’re at the border we don’t go out-of-bounds.
                        crop = image[i:min(i+target_size, height), j:min(j+target_size, width)]
                        crop_mask = mask[i:min(i+target_size, height), j:min(j+target_size, width)]
                        crop_h, crop_w = crop.shape[:2]
                        
                        # If the tile is smaller than target_size in height or width, pad on the bottom/right.
                        pad_bottom = target_size - crop_h if crop_h < target_size else 0
                        pad_right  = target_size - crop_w if crop_w < target_size else 0
                        if pad_bottom > 0 or pad_right > 0:
                            crop = cv2.copyMakeBorder(
                                crop,
                                top=0,
                                bottom=pad_bottom,
                                left=0,
                                right=pad_right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0]
                            )
                            crop_mask = cv2.copyMakeBorder(
                                crop_mask,
                                top=0,
                                bottom=pad_bottom,
                                left=0,
                                right=pad_right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=0
                            )
                        
                        # Save the tile only if its mask contains non-zero pixels.
                        if np.any(crop_mask):
                            # Convert the padded mask to 3 channels.
                            crop_mask_3 = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
                            crop_path = self.output_image_dir / f"{row['cutout_id']}_{i}_{j}.jpg"
                            mask_crop_path = self.output_mask_dir / f"{row['cutout_id']}_{i}_{j}.png"
                            cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                            cv2.imwrite(str(mask_crop_path), crop_mask_3)
            
            else:
                # Otherwise, the entire image is smaller than or equal to target_size.
                # Pad it symmetrically to reach target_size.
                top_pad = (target_size - height) // 2
                bottom_pad = target_size - height - top_pad
                left_pad = (target_size - width) // 2
                right_pad = target_size - width - left_pad
                
                padded_image = cv2.copyMakeBorder(
                    image,
                    top=top_pad,
                    bottom=bottom_pad,
                    left=left_pad,
                    right=right_pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                padded_mask = cv2.copyMakeBorder(
                    mask,
                    top=top_pad,
                    bottom=bottom_pad,
                    left=left_pad,
                    right=right_pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=0
                )
                image_out_path = self.output_image_dir / f"{row['cutout_id']}.jpg"
                mask_out_path = self.output_mask_dir / f"{row['cutout_id']}.png"
                # Convert the padded mask to 3 channels.
                padded_mask_3 = cv2.cvtColor(padded_mask, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(image_out_path), padded_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(str(mask_out_path), padded_mask_3)
                
                
    
    def clean_data_samples(self) -> None:
        from pprint import pprint
        import random
        """If there are more than the predefined number of samples, remove the excess samples starting with the smallest. Use memory size."""
        image_files = sorted(self.output_image_dir.glob("*.jpg"))
        sample_image_files = random.sample(image_files, self.sample_n if len(image_files) > self.sample_n else len(image_files))
        # Unlink all the other files that aren't in the sample_image_files
        for image in image_files:
            if image not in sample_image_files:
                mask = self.output_mask_dir / f"{image.stem}.png"
                image.unlink()
                mask.unlink()

    def create_default_txt(self) -> None:
        """Create default text file."""
        image_files = sorted(self.output_image_dir.glob("*.jpg"))
        
        with open(self.cvat_data_dir / "default.txt", "w") as f:
            for img_file in image_files:
                f.write(f"/{self.image_dir_name}/{img_file.name} {self.mask_dir_name}/{img_file.stem}.png\n")

    
    def create_label_colors_txt(self) -> None:
        """Create label colors text file."""
        with open(self.cvat_data_dir / "label_colors.txt", "w") as f:
            f.write(f"0 0 0 background\n")
            for class_id, common_name in self.unique_mask_values:
                f.write(f"{class_id} {class_id} {class_id} {common_name.replace(' ', '_')}\n")
    
    def zip_cvat_dir(self):
        """Zip CVAT directory."""
        shutil.make_archive(str(self.cvat_data_dir), 'zip', str(self.cvat_data_dir))
    
    def package_data(self):
        """Package data."""
        self.processCutouts()
        self.clean_data_samples()
        self.create_default_txt()
        self.create_label_colors_txt()
        self.zip_cvat_dir()

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg):
    cvat_data_generator = CVATDataGenerator(cfg)
    cvat_data_generator.package_data()

if __name__ == "__main__":
    main()
