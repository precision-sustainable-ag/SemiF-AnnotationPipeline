import pandas as pd
import cv2
from pathlib import Path
from ultralytics import YOLO
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class DetectionProcessor:
    def __init__(self, csv_dir, image_dir, model_path, output_dir, batch_id):
        self.csv_dir = Path(csv_dir)
        self.image_dir = Path(image_dir)
        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
        self.batch_id = batch_id
        self.state = batch_id.split("_")[0]

    def crop_image(self, image, bbox):
        """Crop the image using pixel bounding box coordinates."""
        x_min, y_min, x_max, y_max = bbox
        return image[y_min:y_max, x_min:x_max]

    def classify(self, cropped_image):
        """Classify the cropped image using the YOLO model."""
        result = [x for x in self.model(cropped_image, imgsz=128, stream=True)][0]
        predicted = result.probs.top1
        conf = result.probs.top1conf.cpu().numpy()
        return predicted, conf

    def process_csv(self, csv_file):
        """Process a single CSV file."""
        results = pd.read_csv(csv_file)
        image_name = csv_file.stem + '.jpg'  # Assume images are in JPG format
        image_path = self.image_dir / image_name

        if not image_path.exists():
            logging.warning(f"Image not found: {image_path}")
            return  # Skip if the image does not exist
        
        if results.empty:
            logging.warning(f"No detections found in: {csv_file}")
            output_path = self.output_dir / csv_file.name
            results.to_csv(output_path, index=False)
            return  # Skip if no detections are found

        # Read the image once and get dimensions
        image = cv2.imread(str(image_path))
        height, width, _ = image.shape

        # Add new columns to store classifier predictions
        results['classifier_class'] = None
        results['classifier_classname'] = None
        results['classifier_confidence'] = None

        cropped_images = []
        indices = []

        for idx, row in results.iterrows():
            # Convert normalized bbox coordinates to pixel coordinates
            x_min = int(row['xmin'] * width)
            y_min = int(row['ymin'] * height)
            x_max = int(row['xmax'] * width)
            y_max = int(row['ymax'] * height)

            # Ensure the bounding box coordinates are within the image boundaries
            bbox = (
                max(0, x_min), max(0, y_min),
                min(width, x_max), min(height, y_max)
            )

            cropped = self.crop_image(image, bbox)
            cropped_images.append(cropped)
            indices.append(idx)
        
         # Perform batched inference.
        # Note: Removing 'stream=True' may be necessary so that the model can batch the inputs.
        batch_results = self.model(cropped_images, imgsz=128)


        # Iterate over the results and assign predictions back to the DataFrame
        for idx, result in zip(indices, batch_results):
            # Extract top prediction and its confidence
            prediction = result.probs.top1
            conf = result.probs.top1conf.cpu().numpy()

            results.at[idx, 'classifier_class'] = prediction
            if self.state == "NC":
                results.at[idx, 'classifier_classname'] = "target_weed" if prediction == 1 else "non_target_weed"
            elif self.state == "MD":
                pred_map = {
                    0: "clover",
                    1: "colorchecker",
                    2: "grass",
                    3: "hairy_vetch",
                    4: "horseweed",
                    5: "non_target_weed",
                    6: "winter_pea"
                    }
                results.at[idx, 'classifier_classname'] = pred_map[prediction]
            results.at[idx, 'classifier_confidence'] = conf

        # Save the updated results
        output_path = self.output_dir / csv_file.name
        results.to_csv(output_path, index=False)

    def process_all_csvs(self):
        """Iterate over all CSV files in the directory."""
        csv_files = sorted([x for x in self.csv_dir.glob('*.csv')])
        for csv_file in csv_files:
            log.info(f"Processing {csv_file.name}...")
            self.process_csv(csv_file)

def select_model(cfg: DictConfig) -> Path:
    # Extract state and season from the configuration
    batch_id = cfg.general.batch_id
    state = batch_id.split("_")[0]
    season = cfg.general.season

    # Validate the state
    valid_states = {"NC", "MD", "TX"}
    assert state in valid_states, f"Unknown state: {state} for selecting the correct classifier."

    # Validate the model directory
    model_dir = Path(cfg.data.modeldir) / "nontarget_weed_classifiers"
    assert model_dir.exists(), f"Classifier model directory not found: {model_dir}"

    # Get the model path from the configuration
    try:
        model_path = Path(getattr(cfg.models.nontarget_weed_classifiers, state)[season])
    except AttributeError:
        raise ValueError(f"No valid model path found for state: {state} and season: {season}")

    # Validate the model path
    assert model_path.exists(), f"Classifier model not found: {model_path}"

    return model_path


# Example usage
def main(cfg: DictConfig) -> None:
    batch_id = cfg.general.batch_id
    csv_dir = Path(cfg.batchdata.plant_dects, "merged")
    image_dir = Path(cfg.batchdata.images)
    output_dir = Path(cfg.batchdata.plant_dects, "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = select_model(cfg)

    processor = DetectionProcessor(
        csv_dir=csv_dir,
        image_dir=image_dir,
        model_path=model_path,
        output_dir=output_dir, 
        batch_id=batch_id
    )
    processor.process_all_csvs()
