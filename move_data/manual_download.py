import random
from pathlib import Path

import cv2
import numpy as np


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

    images = sorted([line for line in lines if "images" in line])
    images = sorted([image for image in images if "prediction" not in image])
    images = sorted([image for image in images if "_tmp" not in image])
    metamasks = sorted([line for line in lines if "meta_masks" in line])
    metadata = sorted([line for line in lines if "metadata" in line])
    jsons = sorted([line for line in lines if ".json" in line])
    logs = sorted([line for line in lines if "logs" in line])

dwnlad = images + metamasks + metadata + jsons + logs

batches = set()
for item in dwnlad:
    batch = item.split("/")[0]
    batches.add(batch)

batches_2_download = []
for batch in batches:
    images = batch + "/images"
    meta_masks = batch + "/meta_masks"
    metadata = batch + "/metadata"
    logs = batch + "/logs"
    jsons = batch + "/" + batch + ".json"
    necessary_data = [images, meta_masks, metadata, logs, jsons]
    if set(necessary_data).issubset(dwnlad):
        batches_2_download.append(images)
        batches_2_download.append(meta_masks)
        batches_2_download.append(metadata)
        batches_2_download.append(logs)
        batches_2_download.append(jsons)

batches_2_download = [x for x in batches_2_download if "MD" not in x]
save_path = "/home/psa_images/SemiF-AnnotationPipeline/.batchlogs/batch_download.txt"
with open(save_path, 'w') as f:
    for item in sorted(batches_2_download):
        f.write(f"{item}\n")
