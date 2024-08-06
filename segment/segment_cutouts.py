import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import DictConfig
from skimage import measure
from skimage.segmentation import clear_border

from segment.semif_utils.utils import cutoutmeta2csv

log = logging.getLogger(__name__)

CUTOUT_PROPS = [
    "area",  # float Area of the region i.e. number of pixels of the region scaled by pixel-area.
    "eccentricity",  # float Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    "solidity",  # float Ratio of pixels in the region to pixels of the convex hull image.
    "perimeter",  # float Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
]

class GenCutoutProps:
    def __init__(self, img: np.ndarray, mask: np.ndarray, cutout: np.ndarray):
        self.img = img
        self.mask = mask
        self.cutout = cutout
    
    def from_regprops_table(self, is_primary: bool, colorchecker: bool, connectivity=2) -> Dict:
        """Generate a dictionary of region properties for the cutout mask."""
        labels = measure.label(self.mask, connectivity=connectivity)
        props = measure.regionprops_table(labels, properties=CUTOUT_PROPS)
        
        total_area = np.sum(props['area']) if props['area'].size > 0 else 0
        mean_eccentricity = np.mean(props['eccentricity']) if props['eccentricity'].size > 0 else 0
        mean_solidity = np.mean(props['solidity']) if props['solidity'].size > 0 else 0
        total_perimeter = np.sum(props['perimeter']) if props['perimeter'].size > 0 else 0

        nprops = {
            "area": total_area if not colorchecker else 0.0,
            "eccentricity": mean_eccentricity if not colorchecker else 0.0,
            "solidity": mean_solidity if not colorchecker else 0.0,
            "perimeter": total_perimeter if not colorchecker else 0.0,
            "green_sum": self.calculate_green_sum() if not colorchecker else 0,
            "blur_effect": self.get_blur_effect() if not colorchecker else 0.0,
            "num_components": self.num_connected_components() if not colorchecker else 0
        }

        rgb_mean, rgb_std = self.analyze_image(self.img)
        nprops["cropout_rgb_mean"] = rgb_mean
        nprops["cropout_rgb_std"] = rgb_std
        nprops["is_primary"] = is_primary
        nprops["extends_border"] = self.get_extends_borders()
        return nprops

    def num_connected_components(self) -> int:
        """Calculate the number of connected components in the mask."""
        if len(self.mask.shape) > 2:
            mask = self.mask[..., 0]
        else:
            mask = self.mask
        _, num = measure.label(mask, background=0, connectivity=2, return_num=True)
        return num

    def analyze_image(self, rgb_image: np.ndarray) -> Tuple[List[float], List[float]]:
        """Analyze the RGB image to calculate mean and standard deviation."""
        r, g, b = cv2.split(rgb_image)

        # Create a mask to identify non-black pixels
        mask = None

        # Check if there are any non-zero pixels
        if mask is not None and np.count_nonzero(mask) == 0:
            return self.all_zero_props()

        # Calculate statistics with safety checks
        b_mean, b_std = (0, 0) if not b.size else cv2.meanStdDev(b, mask=mask)
        g_mean, g_std = (0, 0) if not g.size else cv2.meanStdDev(g, mask=mask)
        r_mean, r_std = (0, 0) if not r.size else cv2.meanStdDev(r, mask=mask)

        rgb_mean = [float(r_mean[0][0]), float(g_mean[0][0]), float(b_mean[0][0])]
        rgb_std = [float(r_std[0][0]), float(g_std[0][0]), float(b_std[0][0])]

        return rgb_mean, rgb_std
        
    def calculate_green_sum(self):
        """Returns binary mask if values are within certain "green" HSV range."""
        hsv = cv2.cvtColor(self.cutout, cv2.COLOR_RGB2HSV)
        lower = np.array([40, 70, 120])
        upper = np.array([90, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        # Sum up the number of non-zero pixels (i.e., green pixels)
        green_sum = np.count_nonzero(hsv_mask)
        return green_sum

    def get_blur_effect(self) -> float:
        """Calculate the blur effect of the cutout."""
        if self.cutout.size == 0:
            return 0.0

        try:
            # Calculate the blur effect
            blur_effect = measure.blur_effect(self.cutout, channel_axis=2)
            
            # Check if blur_effect is NaN or if any divide-by-zero might occur
            if np.isnan(blur_effect) or np.isinf(blur_effect):
                return 0.0  # or another default value

            return blur_effect

        except RuntimeWarning as e:
            print(f"RuntimeWarning encountered: {e}")
            return 0.0
    

    def get_extends_borders(self) -> bool:
        """Check if the cutout extends to the borders of the mask."""
        return not np.array_equal(self.mask, clear_border(self.mask))

    def all_zero_props(self) -> Tuple[List[float], List[float]]:
        """Handle the case where all pixel values are zero."""
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


class CutoutProcessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.species_info = self.load_json(cfg.data.species)["species"]
        self.json_files = sorted([x for x in Path(cfg.batchdata.metadata).glob("*.json")])
        self.cutout_dir = Path(self.cfg.batchdata.cutouts)
        self.cutout_dir.mkdir(exist_ok=True, parents=True)

    def load_json(self, file_path: str) -> Dict:
        """Load a JSON file and return its contents as a dictionary."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def load_image(self, file_path: str) -> np.ndarray:
        """Load an image file using OpenCV and return it as a NumPy array."""
        return cv2.imread(file_path)

    def load_mask(self, file_path: str) -> np.ndarray:
        """Load a mask file using OpenCV and return it as a NumPy array."""
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def process_image(self, json_file: Path) -> None:
        """Process a single image and generate cutouts based on annotations."""
        data = self.load_json(json_file)
        image_id = data['image_id']
        image_path = self.get_corresponding_image_file(image_id, mask=False)
        mask_path = self.get_corresponding_image_file(image_id, mask=True)

        if image_path.exists() and mask_path.exists():
            image = self.load_image(str(image_path))
            mask = self.load_mask(str(mask_path))
            self.generate_and_save_products(image, mask, data, image_id)

    def process_images_concurrently(self) -> None:
        """Process all images using multiprocessing."""
        max_workers = int(len(os.sched_getaffinity(0)) / 5)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_image, json_file): json_file for json_file in self.json_files}
            for future in tqdm(as_completed(futures), total=len(self.json_files), desc="Processing Images Concurrently"):
                try:
                    future.result()  # Retrieve result to catch exceptions
                except Exception as e:
                    log.error(f"Error processing file {futures[future]}: {e}")

        

    def process_images_sequentially(self) -> None:
        """Process all images sequentially."""
        for json_file in tqdm(self.json_files, total=len(self.json_files), desc="Processing Images Sequentially"):
            try:
                self.process_image(json_file)
            except Exception as e:
                log.exception(f"Error processing file {json_file}: {e}")
        log.info(f"Condensing cutout results into a single csv file.")
        
        cutout_dir = self.cutout_dir
        batch_id = self.cfg.general.batch_id
        csv_path = Path(cutout_dir, batch_id, f"{batch_id}.csv")
        cutoutmeta2csv(self.cutout_dir, batch_id, csv_path, save_df=True)

    def generate_and_save_products(self, image: np.ndarray, mask: np.ndarray, metadata: List[Dict], image_id: str) -> None:
        annotations = metadata['annotations']
        """Generate and save cutouts, cutout masks, and bounding box crops."""
        for i, annotation in enumerate(annotations):
            if annotation['cutout_exists']:
                bbox = annotation['bbox_xywh']
                x, y, w, h = bbox
                mask_segment = mask[y:y+h, x:x+w]
                image_crop = image[y:y+h, x:x+w]
                class_id = annotation['category_class_id']
                cutout_id = annotation['cutout_id']

                # Generate the cutout and cutout mask
                cutout, cutout_mask = self.extract_cutout(image_crop, mask_segment, class_id)
                h, w = cutout_mask.shape[:2]
                self.save_cutout(cutout_id, cutout, cutout_mask, image_crop)
                annotation["cutout_width"] = w
                annotation["cutout_height"] = h

                # Calculate properties
                cutout_props = GenCutoutProps(image_crop, mask_segment, cutout)
                properties = cutout_props.from_regprops_table(annotation["is_primary"], colorchecker=True if class_id == 28 else False)
                
                
                cutout_metadata = self.format_metadata(metadata, annotation, properties)
                # Save properties to a JSON file
                self.save_metadata(cutout_metadata)

    def get_corresponding_image_file(self, image_id: str, mask=False) -> Path:
        """Find the corresponding file from a list using the image_id."""
        if mask:
            return Path(self.cfg.batchdata.meta_masks, "semantic_masks", f"{image_id}.png")
        else:
            return Path(self.cfg.batchdata.images, f"{image_id}.jpg")

    def extract_cutout(self, image_segment: np.ndarray, mask_segment: np.ndarray, class_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the original color segment and mask from the image using the mask."""
        # Create a mask for the specific class_id
        instance_mask = (mask_segment == class_id)

        cutout_array = image_segment.copy()
        cutout_array[np.where(mask_segment == 0)] = 0

        # Create a mask image with pixel values set to the class_id
        cutout_mask_array = np.where(instance_mask, class_id, 0).astype(np.uint8)

        return cutout_array, cutout_mask_array

    def format_metadata(self, metadata, annotation, properties):
        category_dict = [
            category_dict for category_dict in self.species_info.values()
            if category_dict["class_id"] == annotation["category_class_id"]][0]
        
        cutout_metadata = {
            "season": self.cfg.general.season,
            "datetime": metadata["exif_meta"]["DateTime"],
            "batch_id": metadata["batch_id"],
            "image_id": metadata["image_id"],
            "cutout_id": annotation["cutout_id"],
            "cutout_num": annotation["cutout_id"].split("_")[-1],
            "cutout_props": properties,
            "cutout_height": annotation["cutout_height"],
            "cutout_width": annotation["cutout_width"],
            "category": category_dict
            }
        return cutout_metadata
        

    def save_cutout(self, cutout_id: str, cutout: np.ndarray, cutout_mask: np.ndarray, image_crop: np.ndarray) -> None:
        """Save the cutout, cutout mask, and cropout."""
        cv2.imwrite(str(self.cutout_dir / f'{cutout_id}.png'), cutout, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(str(self.cutout_dir / f'{cutout_id}_mask.png'), cutout_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(str(self.cutout_dir / f'{cutout_id}.jpg'), image_crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    

    def save_metadata(self, cutout_metadata: Dict) -> None:
        """Save the cutout metadata to a JSON file."""
        cutout_id = cutout_metadata["cutout_id"]
        metadata_file = self.cutout_dir / f'{cutout_id}.json'
        with open(metadata_file, 'w') as f:
            json.dump(cutout_metadata, f, indent=4, default=str)

def main(cfg: DictConfig) -> None:
    """
    Main function to execute the vegetation segmentation pipeline.

    Args:
        cfg (DictConfig): Configuration object containing dataset and processing settings.
    """
    processor = CutoutProcessor(cfg)
    
    if cfg.segment.cutouts.multiprocess:
        processor.process_images_concurrently()
    else:
        processor.process_images_sequentially()

    cutout_dir = Path(cfg.data.cutoutdir)
    batch_id = cfg.general.batch_id
    csv_path = Path(cutout_dir, batch_id, f"{batch_id}.csv")
    cutoutmeta2csv(cutout_dir, batch_id, csv_path, save_df=True)