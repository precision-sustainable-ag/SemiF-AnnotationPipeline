import os
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import pandas as pd


class ParseXML:

    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.create_image_list()

    def create_image_list(self):

        images = glob(os.path.join(self.image_path, "*.jpg"))
        image_ids = [
            image.split(os.path.sep)[-1].split(".")[0] for image in images
        ]

        self.image_list = [{
            "id": image_id,
            "path": path
        } for image_id, path in zip(image_ids, images)]

    def parse(self, xml_file):

        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        for i, items in enumerate(root.findall("object")):
            labels = []
            for item in items.findall("name"):
                label = f"target {item.text}"
                labels.append(label)

            for j, item in enumerate(items.findall("bndbox")):
                xmin = float(item.findall("xmin")[0].text)
                xmax = float(item.findall("xmax")[0].text)
                ymin = float(item.findall("ymin")[0].text)
                ymax = float(item.findall("ymax")[0].text)

                top_right = [xmax, ymin]
                bottom_left = [xmin, ymax]
                top_left = [xmin, ymin]
                bottom_right = [xmax, ymax]

                bbox = {
                    "id": str(i),
                    "top_left": top_left,
                    "top_right": top_right,
                    "bottom_left": bottom_left,
                    "bottom_right": bottom_right,
                    "cls": labels[j],
                    "is_normalized": False
                }
                boxes.append(bbox)
        return boxes

    def create_bboxes(self):

        bounding_boxes = dict()

        for image in self.image_list:
            image_id = image["id"]
            xml_file = os.path.join(self.label_path, image_id + ".xml")

            if os.path.exists(xml_file):
                bboxes = self.parse(xml_file)
            else:
                bboxes = []
            bounding_boxes[image_id] = bboxes

        return bounding_boxes

    def __call__(self, *args, **kwargs):

        bounding_boxes = self.create_bboxes()
        return self.image_list, bounding_boxes


class ParseYOLOCsv:

    def __init__(self, image_path, label_path, fullres_image_path=None):
        self.image_path = image_path
        self.label_path = label_path
        if fullres_image_path is not None:
            self.fullres_image_path = fullres_image_path
        else:
             self.fullres_image_path = self.image_path
        self.create_image_list()

    def create_image_list(self):

        images = Path(self.image_path).glob("*.jpg")
        images = [x for x in images]
        fullres_images = Path(self.fullres_image_path).glob("*.jpg")
        fullres_images = [x for x in fullres_images]
        assert len(fullres_images) == len(images)

        image_ids = [x.stem for x in images]

        self.image_list = [{
            "id": image_id,
            "path": str(path),
            "fullres_path": fullres_path
        } for image_id, path, fullres_path in zip(image_ids, images, fullres_images)]

    def parse(self, df):

        boxes = []

        for i, line in df.iterrows():
            cls = line["name"]
            top_left = line["xmin"], line["ymin"]
            top_right = line["xmax"], line["ymin"]
            bottom_left = line["xmin"], line["ymax"]
            bottom_right = line["xmax"], line["ymax"]

            bbox = {
                "id": str(i),
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right,
                "cls": cls,
                "is_normalized": True
            }

            boxes.append(bbox)

        return boxes

    def create_bboxes(self):

        bounding_boxes = dict()
        df = pd.read_csv(self.label_path)

        for image in self.image_list:
            image_id = image["id"]
            _df = df[df["imgname"] == image_id + ".jpg"].reset_index(
                drop=True, inplace=False)

            bboxes = self.parse(_df)
            bounding_boxes[image_id] = bboxes

        return bounding_boxes

    def __call__(self, *args, **kwargs):

        bounding_boxes = self.create_bboxes()

        return self.image_list, bounding_boxes
