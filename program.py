import cv2
import matplotlib.pyplot as plt
import numpy as np
from mongoengine import DynamicDocument, connect

from ExtractComponents import ContourCollection, ExtractCutouts

# TODO make excutable from shell args parser

# Connect to mongoDB collection and create Dynamic collection class
connect(db="opencv2021", host="localhost", port=27017)


class Image(DynamicDocument):
    meta = {"collection": "image"}


# Define where masks are located
mask_dir = "data/test_results"

# Iterate mongodb collection documents
for img in Image.objects:
    # Extract components
    ecut = ExtractCutouts(mask_dir, img)
    # Assign document fields
    cutout_list_obj = ContourCollection(mask_dir, img)

    # Uncomment to assign/save/update
    # img.cutouts = cutout_list_obj.cutout_list
    # img.update(unset__vegetation_cutouts=True)
    # img.save()

    # Extract contours from database and view
    for cutout in img.cutouts:
        orig_img = plt.imread(f"data/sample/{img.file_name}")
        arr_cntr = np.array(cutout["contours"], dtype=np.int32)
        cv2.drawContours(orig_img, np.array([arr_cntr], dtype=np.int32), -1,
                         (0, 0, 255), 10)
        plt.imshow(orig_img)
        plt.show()
