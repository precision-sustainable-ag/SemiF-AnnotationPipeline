import cv2
import pandas as pd
import os
from pathlib import Path
from typing import List
import hydra
from omegaconf import DictConfig
from update_and_move import BatchDataProcessor


class ImageReviewer:
    def __init__(self, image_dirs: List[str], csv_file_name: str):
        """
        Initializes the Image Reviewer.

        :param image_dirs: List of directories to scan for images.
        :param csv_file: CSV file to store review results.
        """
        self.image_dirs = image_dirs
        self.csv_file = Path(image_dirs[0].parent, csv_file_name)
        self.df = self.load_existing_reviews()
        self.processed_images = set(zip(self.df["Image Name"], self.df["Directory"]))
        self.new_entries = []

    def load_existing_reviews(self) -> pd.DataFrame:
        """Loads existing reviews from the CSV file if it exists."""
        if os.path.exists(self.csv_file):
            return pd.read_csv(self.csv_file)
        return pd.DataFrame(columns=["Image Name", "Directory", "Status"])

    def review_images(self):
        """Iterates over directories and reviews images one by one."""
        for image_dir in self.image_dirs:
            directory_name = Path(image_dir).name  # Extract folder name
            
            if "cutout" in directory_name:
                for sub_dir in image_dir.iterdir():
                    cutout_files = sorted(sub_dir.glob("*.png"))
                    for cutout_path in cutout_files:
                        cutout_name = cutout_path.name

                        # Skip if already reviewed
                        if (cutout_name, directory_name) in self.processed_images:
                            continue

                        self.show_image(cutout_path, sub_dir.name)
            else:
                image_files = sorted(Path(image_dir).glob("*.jpg"))  # Adjust for other formats if needed
                
                for image_path in image_files:
                    image_name = image_path.name

                    # Skip if already reviewed
                    if (image_name, directory_name) in self.processed_images:
                        continue

                    self.show_image(image_path, directory_name)

        # Close the OpenCV window after the last image is processed
        cv2.destroyAllWindows()

    def show_image(self, image_path: Path, directory_name: str):
        """Displays an image and waits for user input."""
        img = cv2.imread(str(image_path))

        if img is None:
            print(f"Error loading image: {image_path.name}")
            return

        # Resize image: Reduce width by 20% while maintaining aspect ratio
        height, width = img.shape[:2]
        if "mask" in directory_name:
            new_width = int(width * 0.8)
            new_height = int((new_width / width) * height)
            resized_img = cv2.resize(img, (new_width, new_height))

        if "bbox" in directory_name:
            new_width = int(width * 0.7)
            new_height = int((new_width / width) * height)
            resized_img = cv2.resize(img, (new_width, new_height))

        else:
            new_width = int(width * 0.7)
            new_height = int((new_width / width) * height)
            resized_img = cv2.resize(img, (new_width, new_height))

        cv2.imshow("Image Review", resized_img)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("a"):  # Pass
                self.new_entries.append([image_path.name, directory_name, "pass"])
                print(f"Marked {image_path.name} as 'pass'.")
                self.save_results()
                break

            elif key == ord("s"):  # Fail
                self.new_entries.append([image_path.name, directory_name, "fail"])
                print(f"Marked {image_path.name} as 'fail'.")
                self.save_results()
                break

            elif key == ord("q"):  # Quit and save progress
                print("Exiting review.")
                cv2.destroyAllWindows()
                self.save_results()
                exit()

            else:
                print("Invalid key. Press 'p' (pass), 's' (fail), or 'q' (quit).")
    
    def save_results(self):
        """Saves results to CSV file."""
        if self.new_entries:
            new_df = pd.DataFrame(self.new_entries, columns=["Image Name", "Directory", "Status"])
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df.to_csv(self.csv_file, index=False)
            print(f"Results saved to {self.csv_file}.")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Example: Add multiple directories here
    batch_id = cfg.general.batch_id
    validation_dir = Path(cfg.general.workdir, "validate", "results",batch_id)
    
    directories_to_review = [
        validation_dir / "bboxes", 
        validation_dir / "masks", 
        validation_dir / "cutouts"
        ]
    
    reviewer = ImageReviewer(directories_to_review, csv_file_name=f"{batch_id}.csv")
    reviewer.review_images()
    df = reviewer.df

    # If there are any failed images exit the script
    if "fail" in df["Status"].values:
        print("Validation failed. Please review the images and try again.")
        exit()
    else:
        print("Validation successful. Proceeding to update and move images.")
        processor = BatchDataProcessor(cfg)
        processor.process()




if __name__ == "__main__":
    main()
