import matplotlib.pyplot as plt
import json


def display_images(images, rows=1, cols=1, figsize=(12, 8)):
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for ind, img in enumerate(images):
        ax.ravel()[ind].imshow(img)
        # ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()


def read_bbox_from_json(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())

    return y1, y2, x1, x2