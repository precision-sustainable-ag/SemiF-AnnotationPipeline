import cv2
import pandas as pd
import os
from pathlib import Path
from typing import List
import hydra
import numpy as np
from omegaconf import DictConfig
from update_and_move import BatchDataProcessor
from validate_utils import batch_df, validation_sample_df, get_bboxes_validation_images, get_mask_validation_images, get_cutout_validate_images

class ImageReviewer:
    def __init__(self,cfg: DictConfig, batch_dir: Path, df: pd.DataFrame, image_dirs: List[str], csv_file_name: str):
        """
        Initializes the Image Reviewer.

        :param image_dirs: List of directories to scan for images.
        :param csv_file: CSV file to store review results.
        """
        self.species_info = cfg.data.species
        self.prcossed_df = df
        self.image_dirs = image_dirs
        self.csv_file = Path(batch_dir, csv_file_name)
        self.df = self.load_existing_reviews()
        self.processed_images = set(self.df["image_id"])
        self.new_entries = []
        
        # create a column in self.processed_df to store the product type and get the values from the csv file
        if "product" in self.df.columns:
            product_map = self.df.groupby("image_id")["product"].apply(list).to_dict()
            from pprint import pprint
            self.prcossed_df["product"] = self.prcossed_df["image_id"].map(product_map)

    def load_existing_reviews(self) -> pd.DataFrame:
        """Loads existing reviews from the CSV file if it exists."""
        if self.csv_file.exists():
            return pd.read_csv(self.csv_file)
        df_columns = list(self.prcossed_df.columns) + ["Status", "product"]
        return pd.DataFrame(columns=df_columns)

    def review_images(self):
        """Iterates over directories and reviews images one by one."""
        
        # print the possible key options
        print("\nPossible keys options:")
        print("'a' to pass")
        print("'s' to fail")
        print("'t' to tag for further review")
        print("'q' to quit\n")

        annotated_images = get_bboxes_validation_images(self.prcossed_df, self.processed_images)
        for image, row in annotated_images:
            self.show_image(image, row, product="bbox")
        cv2.destroyAllWindows()
        
        mask_images = get_mask_validation_images(
            self.prcossed_df, 
            self.processed_images,
            species_info=self.species_info,
            resize_factor=0.8
            )
        for image, row in mask_images:
            self.show_image(image, row, product="mask")
        # cv2.destroyAllWindows()

        cutout_images = get_cutout_validate_images(
            self.prcossed_df, 
            self.processed_images,
            resize_factor=0.8
            )
        for image, row in cutout_images:
            self.show_image(image, row, product="cutout")
        cv2.destroyAllWindows()

    def show_image(self, img: np.ndarray, row: pd.Series, product: str):
        """Displays an image and waits for user input."""
       
        # Resize image: Reduce width by 20% while maintaining aspect ratio
        height, width = img.shape[:2]
        if product == "mask":
            new_width = int(width * 0.15)
        
        elif product == "bbox":
            new_width = int(width * 0.7)

        elif product == "cutout":
            new_width = int(width * 0.9)
        
        new_height = int((new_width / width) * height)
        resized_img = cv2.resize(img, (new_width, new_height))

        cv2.imshow("Image Review", resized_img)

        image_id = row["image_id"]

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("a"):  # Pass
                row["Status"] = "pass"
                row["product"] = product
                # add the entire row plus "status" to the new_entries list

                self.new_entries.append(row)
                self.save_results()
                break

            elif key == ord("s"):  # Fail
                row["Status"] = "fail"
                row["product"] = product
                self.new_entries.append(row)
                self.save_results()
                break

            elif key == ord("t"):  # Tag for further review
                row["Status"] = "review"
                row["product"] = product
                self.new_entries.append(row)
                self.save_results()
                break

            elif key == ord("q"):  # Quit and save progress
                print("Exiting review.")
                cv2.destroyAllWindows()
                self.save_results()
                exit()

            else:
                print("Invalid key. Press 'p' (pass), 's' (fail), 't' (review),a or 'q' (quit).")
    
    
    def save_results(self):
        """Saves results to CSV file."""
        if self.new_entries:
            new_df = pd.DataFrame(self.new_entries)
            if not self.df.empty:
                # Ensure we update the correct rows instead of overwriting everything
                self.df = pd.concat([self.df, new_df], ignore_index=True)
            else:
                self.df = new_df
            # Remove duplicates based on "image_id" and "product" (latest decision will be kept)
            self.df = self.df.drop_duplicates(subset=["image_id", "product"], keep="last")
            # Save to CSV
            self.df.to_csv(self.csv_file, index=False)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # 
    
    # Example: Add multiple directories here
    batch_id = cfg.general.batch_id
    validation_dir = Path(cfg.general.workdir, "validate", "results",batch_id)
    ogdf = batch_df(batch_id, cfg.data.cutoutdir, cfg.data.batchdir)
    df = validation_sample_df(ogdf, sample_sz=cfg.validate.sample_sz, random_state=cfg.validate.random_state)
    
    directories_to_review = [
        validation_dir / "bboxes", 
        validation_dir / "masks", 
        validation_dir / "cutouts"
        ]
    
    reviewer = ImageReviewer(cfg, Path(cfg.data.batchdir), df, directories_to_review, csv_file_name=f"{batch_id}_validation_results.csv")
    reviewer.review_images()
    df = reviewer.df

    # If there are any failed images or images that need further review, exit the script
    if "fail" in df["Status"].values:
        print("Validation failed. Please review the images and try again.")
        exit()
    
    elif "review" in df["Status"].values:
        print("Some images have been tagged for further review. Please review the images and try again.")
    
    else:
        print("Validation successful. Proceeding to update and move images.")
        processor = BatchDataProcessor(cfg)
        processor.process()




if __name__ == "__main__":
    main()
