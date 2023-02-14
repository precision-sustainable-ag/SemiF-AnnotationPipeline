import cv2
import numpy as np
from pathlib import Path
import random


class Test:
    ALL_CLASSES = ('background', 'Palmer amaranth', 'Common ragweed',
                   'Sicklepod', 'Cocklebur', 'Large crabgrass', 'Goosegrass',
                   'Broadleaf signalgrass', 'Purple nutsedge', 'Waterhemp',
                   'Barnyardgrass', 'Jungle rice', 'Texas millet', 'Kochia',
                   'Common sunflower', 'Ragweed parthenium', 'Johnsongrass',
                   'Soybean', 'Smooth pigweed', 'Common lambsquarters',
                   'Fall panicum', 'Jimson weed', 'Velvetleaf',
                   'Yellow foxtail', 'Giant foxtail', 'Horseweed', 'Maize',
                   'unknown', 'colorchecker', 'Hairy vetch', 'Winter pea',
                   'Crimson clover', 'Red clover', 'Mustards',
                   'cultivated radish', 'Cereal rye', 'Triticale',
                   'Winter wheat', 'Oats', 'Barley', 'Black oats')

    CLASSES = ('background', 'Palmer amaranth', 'Common ragweed', 'Sicklepod',
               'Cocklebur', 'Large crabgrass', 'Goosegrass',
               'Broadleaf signalgrass', 'Purple nutsedge', 'Waterhemp',
               'Barnyardgrass', 'Jungle rice', 'Texas millet', 'Kochia',
               'Common sunflower', 'Ragweed parthenium', 'Johnsongrass',
               'Soybean', 'Smooth pigweed', 'Common lambsquarters',
               'Fall panicum', 'Jimson weed', 'Velvetleaf', 'Yellow foxtail',
               'Giant foxtail', 'Horseweed', 'Maize', 'unknown',
               'colorchecker', 'Hairy vetch', 'Winter pea', 'Crimson clover',
               'Red clover', 'Mustards', 'cultivated radish', 'Cereal rye',
               'Triticale', 'Winter wheat', 'Oats', 'Barley', 'Black oats')

    def __init__(self):

        # print(property_asel)
        class_indices = [
            list(self.ALL_CLASSES).index(x) if x != 'background' else 0
            for x in self.CLASSES
        ]
        print(class_indices)
        self.label_map = dict(
            zip([x for x in class_indices],
                [1 if x != 0 else 0 for x in range(0, 150)]))
        print(self.label_map)
        # print(self.label_map)


import random

# path = "/home/psa_images/SemiF-AnnotationPipeline/.batchlogs/ordered_batches.txt"
# with open(path, 'r') as f:
#     lines = [line.rstrip() for line in f]
#     # Strip azcopy list results strings
#     lines = [x.replace("INFO: ", "") for x in lines]
#     lines = [x.split(";")[0] for x in lines]
#     imgs = [
#         Path(x) for x in lines if not x.endswith(".ARW") and (
#             x.endswith(".JPG") or x.endswith(".jpg")) and "prediction" not in x
#     ]
#     imgnames = sorted(imgs)

# random.seed(42)
# rand_imgs = random.sample(imgnames, 100)

# save_path = "/home/psa_images/SemiF-AnnotationPipeline/.batchlogs/random_field_names.txt"
# with open(save_path, 'w') as f:
#     for imgn in rand_imgs:
#         f.write(f"{imgn}\n")

path = "/home/psa_images/SemiF-AnnotationPipeline/.batchlogs/container_list.txt"
with open(path, 'r') as f:
    lines = [line.rstrip() for line in f]

    images = [line for line in lines if "images" in line]
    images = [image for image in images if "prediction" not in image]
    images = [image for image in images if "_tmp" not in image]
    metamasks = [line for line in lines if "meta_masks" in line]
    metadata = [line for line in lines if "metadata" in line]
    jsons = [line for line in lines if ".json" in line]
    logs = [line for line in lines if "logs" in line]
    # download = [[
    #     img, mm, metad, json, log
    # ] for img, mm, metad, json, log in zip(sorted(images), sorted(
    #     metamasks), sorted(metadata), sorted(jsons), sorted(logs))]

dwnlad = images + metamasks + metadata + jsons + logs
save_path = "/home/psa_images/SemiF-AnnotationPipeline/.batchlogs/batch_download.txt"
with open(save_path, 'w') as f:
    for item in dwnlad:
        f.write(f"{item}\n")