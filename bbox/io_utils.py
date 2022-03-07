import os
from glob import glob
import xml.etree.ElementTree as ET


class ParseXML:

    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.create_image_list()

    def create_image_list(self):

        images = glob(os.path.join(self.image_path, "*.jpg"))
        image_ids = [image.split(os.path.sep)[-1].split(".")[0] for image in images]

        self.image_list = [{"id": image_id, "path": path} for image_id, path in zip(image_ids, images)]

    def parse(self, xml_file):
        # print(xml_file)

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
                    "cls": labels[j]
                }
                boxes.append(bbox)
        return boxes

    def create_bboxes(self):

        bounding_boxes = dict()
        
        for image in self.image_list:
            image_id = image["id"]
            xml_file = os.path.join(self.label_path, image_id+".xml")
            
            if os.path.exists(xml_file):
                bboxes = self.parse(xml_file)
            else:
                bboxes = []
            bounding_boxes[image_id] = bboxes

        return bounding_boxes

    def __call__(self, *args, **kwargs):

        bounding_boxes = self.create_bboxes()
        return self.image_list, bounding_boxes
