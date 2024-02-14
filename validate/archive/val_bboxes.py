import glob
import json
import math
import os
import shutil


class JsonFileHandler:
    def __init__(
        self, filepath, backup_dir=None, corrected_dir=None, save_to_original=False
    ):
        self.save_to_original = save_to_original
        self.filepath = filepath
        self.backup_dir = backup_dir
        self.corrected_dir = corrected_dir
        self.data = self._load_json()

    def _load_json(self):
        with open(self.filepath, "r") as file:
            return json.load(file)

    def save_corrected_json(self, data):
        if self.backup_dir:
            self.backup_original()

        if self.corrected_dir:
            filepath = os.path.join(self.corrected_dir, os.path.basename(self.filepath))
            assert (
                self.save_to_original is False
            ), "save_to_original keyword argument is True and corrected_dir is set."
        else:
            assert (
                self.save_to_original is True
            ), "Trying to save to the original json path without setting save_to_original keyword argument. Are you sure you want to save?"
            filepath = self.filepath

        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)  # Using indent=4 for pretty printing

    def backup_original(self):
        backup_filepath = os.path.join(self.backup_dir, os.path.basename(self.filepath))
        shutil.copy2(self.filepath, backup_filepath)

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data


class BBoxComparator:
    def __init__(self, batch_ids):
        self.batch_ids = batch_ids
        self.cutout_dir = (
            "/mnt/research-projects/s/screberg/longterm_images/semifield-cutouts/"
        )
        self.dev_dir = "/mnt/research-projects/s/screberg/longterm_images/semifield-developed-images"

    def load_json(self, filepath):
        """Load a JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def get_cutout_file_path(self, bbox_id, cutout_dir_path):
        return os.path.join(cutout_dir_path, f"{bbox_id}.json")

    def process_dev_bbox(self, bbox, img_width, img_height):
        """Process a bbox from dev_data."""
        bbox = self.ensure_normalized(bbox, img_width, img_height)
        return bbox

    def process_cutout_bbox(self, bbox, img_width, img_height):
        # check if it's a list, if so create format
        bbox = self.ensure_bbox_dict_format(bbox)

        # check if it's normalized, if not, make it normalized
        bbox = self.ensure_normalized(bbox, img_width, img_height, dev=False)

        return bbox

    def compare_bboxes(self, dev_bbox, cutout_bbox):
        # Assuming local_coordinates is the key we want to compare
        dev_coords = dev_bbox.get("local_coordinates", {})

        cutout_coords = cutout_bbox  # cutout_bbox.get("local_coordinates", {})

        return dev_coords == cutout_coords

    @staticmethod
    def is_bbox_list_format(bbox):
        return isinstance(bbox, list) and len(bbox) == 4

    @staticmethod
    def ensure_coordinates_are_ints(bbox):
        """Convert coordinates in a bbox to integers."""
        for key, value in bbox["local_coordinates"].items():
            # print(key, value)

            if key in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                bbox["local_coordinates"][key] = [int(coord) for coord in value]
        return bbox

    @staticmethod
    def convert_bbox_to_dict_format(bbox_list):
        x1, y1, x2, y2 = bbox_list
        bbox_dict = {
            "top_left": [x1, y1],
            "top_right": [x2, y1],
            "bottom_left": [x1, y2],
            "bottom_right": [x2, y2],
            "is_scaleable": True,
        }
        return bbox_dict

    @staticmethod
    def are_coordinates_normalized(self, x, y):
        return 0 <= x <= 1 and 0 <= y <= 1

    @staticmethod
    def normalize_coordinates(bbox, img_width, img_height):
        """Normalize bbox coordinates with respect to image width and height."""
        for key, value in bbox.items():
            if key in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                x, y = value
                normalized_x = round(x / img_width, 6)
                normalized_y = round(y / img_height, 6)
                bbox[key] = [normalized_x, normalized_y]
        return bbox

    def ensure_normalized(self, bbox, img_width, img_height, dev=True):
        """Ensure that the bbox coordinates are normalized."""
        if dev:
            for key, value in bbox["local_coordinates"].items():
                if key in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                    x, y = bbox["local_coordinates"].get(key, [0, 0])

                    if not (0 <= value[0] <= 1 and 0 <= value[1] <= 1):
                        if value[0] < 0 or value[1] < 0:
                            # Ensure non-negative values
                            normalized_x = max(0, normalized_x)
                            normalized_y = max(0, normalized_y)
                        bbox["local_coordinates"] = self.normalize_coordinates(
                            bbox["local_coordinates"], img_width, img_height
                        )
                        break
        else:
            for key, value in bbox.items():
                if key in ["top_left", "top_right", "bottom_left", "bottom_right"]:
                    if not (0 <= value[0] <= 1 and 0 <= value[1] <= 1):
                        bbox = self.normalize_coordinates(bbox, img_width, img_height)
                        break

        return bbox

    def ensure_bbox_dict_format(self, bbox):
        if self.is_bbox_list_format(bbox):
            return self.convert_bbox_to_dict_format(bbox)
        return bbox

    def check_cutout_bbox(self, dev_bbox, cutout_file):
        """Check if the cutout bbox matches with the dev bbox."""
        cutout_data = self.load_json(cutout_file)
        if "bbox" in cutout_data:
            cutout_bbox = self.ensure_bbox_dict_format(cutout_data["bbox"])
            if not self.compare_bboxes(dev_bbox, cutout_bbox):
                return False
        return True

    @staticmethod
    def compare_bboxes(bbox1, bbox2, tolerance=1e-5, max_difference=0.1):
        """Compare two bounding boxes to see if they are close enough."""
        for key in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            x1, y1 = bbox1.get(key, [0, 0])
            x2, y2 = bbox2.get(key, [0, 0])

            # Check if the coordinates are close enough
            if not (
                math.isclose(x1, x2, rel_tol=tolerance, abs_tol=max_difference)
                and math.isclose(y1, y2, rel_tol=tolerance, abs_tol=max_difference)
            ):
                return False

            # Check if the difference is way off by more than a tenth
            if abs(x1 - x2) > max_difference or abs(y1 - y2) > max_difference:
                print(f"Coordinates for key {key} are way off in comparison!")
                return False

        return True

    # Modify process_batch_id
    def process_batch_id(self, batch_id):
        full_dev_dir_path = os.path.join(self.dev_dir, batch_id, "metadata")
        full_cutout_dir_path = os.path.join(self.cutout_dir, batch_id)

        dev_json_files = glob.glob(os.path.join(full_dev_dir_path, "*.json"))

        counter = 0
        for dev_file in dev_json_files:
            dev_data = self.load_json(dev_file)
            img_width, img_height = dev_data.get("width"), dev_data.get("height")

            for bbox in dev_data.get("bboxes", []):
                processed_dev_bbox = self.process_dev_bbox(bbox, img_width, img_height)
                bbox_id = bbox.get("bbox_id")
                corresponding_cutout_file = self.get_cutout_file_path(
                    bbox_id, full_cutout_dir_path
                )

                if os.path.exists(corresponding_cutout_file):
                    cutout_data = self.load_json(corresponding_cutout_file)
                    cutout_bbox = cutout_data["bbox"]
                    cutout_bbox = self.process_cutout_bbox(
                        cutout_bbox, img_width=img_width, img_height=img_height
                    )

                    # print(processed_dev_bbox["local_coordinates"])
                    # print(cutout_bbox)
                    if not self.compare_bboxes(
                        processed_dev_bbox["local_coordinates"],
                        cutout_bbox,
                        tolerance=0.0000001,
                    ):
                        # if not self.check_cutout_bbox(
                        # processed_bbox, corresponding_cutout_file
                        # ):
                        # print(
                        #     f"Mismatch detected for bbox_id {bbox_id}\n{processed_dev_bbox['local_coordinates']}\n{cutout_bbox}"
                        # )
                        counter += 1
        print(len(list(dev_json_files)))
        print(counter)

    def run(self):
        for batch_id in self.batch_ids:
            self.process_batch_id(batch_id)


class BBoxChecker:
    def __init__(self, data, x_range=(0, 9560), y_range=(0, 6368)):
        # def __init__(self, data, x_range=(0, 1), y_range=(0, 1)):
        self.data = data
        self.x_range = x_range
        self.y_range = y_range

    def are_coordinates_normalized(self, x, y):
        return 0 <= x <= 1 and 0 <= y <= 1

    def normalize_coordinates(self, x, y, image_width, image_height):
        return x / image_width, y / image_height


# Example usage:
batch_ids = [
    # "NC_2023-02-03",
    # "NC_2023-02-06",
    # "NC_2023-02-20",
    # "NC_2023-02-22",
    # "NC_2023-03-07",
    # "NC_2023-06-12",
    # "NC_2023-07-03",
    # "NC_2023-07-10",
    "NC_2023-07-11",
]
comparator = BBoxComparator(batch_ids)
comparator.run()
