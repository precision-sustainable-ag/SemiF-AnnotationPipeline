import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

sys.path.append("/home/psa_images/SemiF-AnnotationPipeline")
sys.path.append("/home/psa_images/SemiF-AnnotationPipeline/segment")
from segment.semif_utils.utils import apply_mask


def read_metadata(path):
    with open(path, "r") as f:
        data = json.loads(f.read())
    return data


def batch_df(batch_id, cutout_dir, batch_dir):
    df = pd.read_csv(Path(cutout_dir, batch_id, batch_id + ".csv"))
    df["state_id"] = df.batch_id.str.split("_", expand=False).str[0]
    df["date"] = df.batch_id.str.split("_", expand=False).str[1]
    df["cutout_paths"] = cutout_dir + "/" + batch_id + "/" + df["cutout_id"] + ".png"
    df["image_paths"] = batch_dir + "/images/" + df["image_id"] + ".jpg"
    df["meta_paths"] = batch_dir + "/metadata/" + df["image_id"] + ".json"
    df["semantic_masks"] = (
        batch_dir + "/meta_masks/semantic_masks/" + df["image_id"] + ".png"
    )
    df["instance_masks"] = (
        batch_dir + "/meta_masks/instance_masks/" + df["image_id"] + ".png"
    )
    return df


def validation_sample_df(df, sample_sz=10, random_state=42, drop_img_duplicates=True):
    if drop_img_duplicates:
        df = df.drop_duplicates(
            subset="image_paths", keep="first"
        )  # Emphasize different full-res images and not cutouts
    df = df.sample(sample_sz, random_state=random_state)
    return df


def get_detection_data(jsonpath):
    meta = read_metadata(jsonpath)
    imgwidth, imgheight = meta["fullres_width"], meta["fullres_height"]
    bboxes = meta["bboxes"]
    boxes = []
    labels = []
    for box in bboxes:
        x1 = box["local_coordinates"]["top_left"][0] * imgwidth
        y1 = box["local_coordinates"]["top_left"][1] * imgheight

        x2 = box["local_coordinates"]["bottom_right"][0] * imgwidth
        y2 = box["local_coordinates"]["bottom_right"][1] * imgheight

        boxes.append([x1, y1, x2, y2])

        if box["cls"] == "plant":
            class_id = box["cls"]
            common_name = box["cls"]
        else:
            class_id = str(box["cls"]["class_id"])
            common_name = box["cls"]["common_name"]

        label = f"{common_name} ({class_id})"
        labels.append(label)

    return boxes, labels


def save_original_full_res_images(
    df,
    save_location=".",
):
    for _, i in df.iterrows():
        src_path = i["image_paths"]
        dst_dir = Path(save_location, "full_res_images")
        dst_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy2(src_path, dst_dir)


def plot_bboxes(
    df,
    show_labels=True,
    transparent_fc=True,
    save_location=".",
    axis=False,
    figsize=(8, 12),
    dpi=300,
):
    """plots bounding boxes for inspections

    Args:
        df (pandas dataframe): cutout csv
        show_labels (bool, optional): show labels. Defaults to True.
        save (bool, optional): save figure . Defaults to False.
        transparent_fc (bool, optional): transparent face or white. Defaults to True.
        save_location (str, optional): save location of plot. Defaults to ".".
        figsize (tuple, optional): figure size. Defaults to (8, 12).
    """
    for _, i in df.iterrows():
        image_path = i["image_paths"]

        i = Path(image_path)
        # cv2.imread(image_path)
        meta_path = image_path.replace(f"images/{i.name}", f"metadata/{i.stem}.json")
        bboxes, labels = get_detection_data(meta_path)
        if len(bboxes) == 0:
            continue
        else:
            fig = plt.figure(
                figsize=figsize, facecolor="none" if transparent_fc else "w"
            )

            # add axes to the image
            ax = fig.add_axes([0, 0, 1, 1])

            # read and plot the image
            # image = plt.imread(image_path)
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)

            if not axis:
                ax.axis(False)

            # Iterate over all the bounding boxes
            linewidth = (fig.get_figwidth() + fig.get_figheight()) * 0.1
            fontsize = (
                fig.get_figwidth() + fig.get_figheight()
            ) * 0.5  # Adjust the scaling factor as desired
            for i, bbox in enumerate(bboxes):
                xmin, ymin, xmax, ymax = bbox
                w = xmax - xmin
                h = ymax - ymin

                # add bounding boxes to the image

                box = patches.Rectangle(
                    (xmin, ymin),
                    w,
                    h,
                    linewidth=linewidth,
                    edgecolor="red",
                    facecolor="none",
                )

                ax.add_patch(box)

                if show_labels and labels is not None:
                    rx, ry = box.get_xy()
                    ax.text(
                        rx,
                        (ry - 130),
                        str(labels[i]),
                        verticalalignment="top",
                        color="white",
                        fontsize=fontsize,
                        weight="bold",
                    )

            save_location = Path(save_location)
            save_location.mkdir(exist_ok=True, parents=True)
            image_stem = Path(image_path).stem + "_bbox_plot"
            plot_path = Path(
                save_location,
                image_stem + ".png" if transparent_fc else image_stem + ".jpg",
            )
            fig.savefig(
                plot_path, bbox_inches="tight", transparent=transparent_fc, dpi=dpi
            )
            plt.close()


def species_info2color_map(species_info):
    data = read_metadata(species_info)
    class_id_to_rgb = {
        item["class_id"]: item["rgb"] for item in data["species"].values()
    }
    return class_id_to_rgb


def convert_mask_values(mask, color_mapping):
    mask = mask[..., 0]

    # Identify the unique non-zero values in the mask
    unique_values = set(np.unique(mask[mask > 0]))

    # Update the color_mapping to only include keys that are in unique_values
    color_mapping = {
        key: value for key, value in color_mapping.items() if key in unique_values
    }

    # Create an empty RGB image with the same shape as the mask
    rows, cols = mask.shape
    colored_mask = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Apply the colors based on the updated mapping
    for value, color in color_mapping.items():
        colored_mask[mask == value] = color

    return colored_mask


def plot_masks(
    df_sample,
    figsize=(8, 12),
    transparent_fc=True,
    include_suptitles=True,
    save_location=".",
    species_info=None,
    dpi=300,
):
    for _, df in df_sample.iterrows():
        imgpath = df["image_paths"]
        maskpath = df["semantic_masks"]
        instancepath = df["instance_masks"]
        bgr = cv2.imread(imgpath)
        rgbimg = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        bgrsemmask = cv2.imread(maskpath)

        rgbmask = cv2.cvtColor(bgrsemmask, cv2.COLOR_BGR2RGB)

        color_map = species_info2color_map(species_info)
        rgbmask = convert_mask_values(rgbmask, color_map)

        bgrinmask = cv2.imread(instancepath)
        rgbinstance = cv2.cvtColor(bgrinmask, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=figsize, facecolor="none" if transparent_fc else "w"
        )

        ax1.imshow(rgbimg)
        ax2.imshow(rgbmask)
        ax3.imshow(rgbinstance)

        ax1.axis(False)
        ax2.axis(False)
        ax3.axis(False)

        if include_suptitles:
            fontsize = (
                fig.get_figwidth() + fig.get_figheight()
            ) * 0.5  # Adjust the scaling factor as desired
            ax1.title.set_text(f"Common name: {df.common_name}")
            ax1.title.set_fontsize(fontsize)
            ax2.title.set_text("Unique semantic mask values (class_id): ")
            ax2.title.set_fontsize(fontsize)
            ax3.title.set_text(f"{np.unique(bgrsemmask[..., 0])}")
            ax3.title.set_fontsize(fontsize)

        plt.tight_layout()
        save_location = Path(save_location)
        save_location.mkdir(exist_ok=True, parents=True)
        image_stem = Path(imgpath).stem + "_mask_plot"
        plot_path = Path(
            save_location,
            image_stem + ".png" if transparent_fc else image_stem + ".jpg",
        )
        fig.savefig(plot_path, bbox_inches="tight", transparent=transparent_fc, dpi=dpi)
        plt.close()


def filter_by_area(df, area_min, area_max):
    common_names = df["common_name"].unique()
    dfs = []
    for common_name in common_names:
        cname_df = df[df["common_name"] == common_name]
        desc = cname_df["area"].describe()
        bounds = {
            "mean": desc.iloc[1],
            25: desc.iloc[4],
            50: desc.iloc[5],
            75: desc.iloc[6],
        }
        min_bound, max_bound = bounds.get(area_min), bounds.get(area_max)

        if min_bound is not None:
            cname_df = cname_df[cname_df["area"] > min_bound]
        if max_bound is not None:
            cname_df = cname_df[cname_df["area"] < max_bound]
        dfs.append(cname_df)

    df = pd.concat(dfs)

    return df


def filter_by_num_components(df, component_min, component_max):
    if component_min != None:
        df = df[df["num_components"] > component_min]

    if component_max != None:
        df = df[df["num_components"] < component_max]

    return df


def filter_by_solidity(df, solid_min, solid_max):
    if solid_min != None:
        df = df[df["solidity"] > solid_min]

    if solid_max != None:
        df = df[df["solidity"] < solid_max]

    return df


def filter_by_species(df, species=None):
    if species:
        df = df[df.USDA_symbol.isin(species)]
    return df


def filter_by_properties(self, df, extends_border=None, is_primary=None):
    if extends_border != "None":
        df = df[df.extends_border == self.extends_border]
    if is_primary != "None":
        df = df[df["is_primary"] == self.is_primary]
    return df


def plot_cutouts(
    df,
    figsize=(8, 12),
    save_location=".",
    transparent_fc=True,
    title=True,
    dpi=300,
):
    mdf = df.copy()

    for species in mdf["common_name"].unique():
        sdf = mdf[mdf["common_name"] == species]

        if len(sdf) == 0:
            continue

        for _, row in sdf.iterrows():
            cutimgp = row["cutout_paths"]
            cropimgp = row["cutout_paths"].replace(".png", ".jpg")
            cutmaskp = row["cutout_paths"].replace(".png", "_mask.png")

            cropimg = cv2.cvtColor(cv2.imread(cropimgp, -1), cv2.COLOR_BGR2RGB)

            cutimg = cv2.cvtColor(cv2.imread(cutimgp, -1), cv2.COLOR_BGR2RGB)

            cutmask = cv2.imread(cutmaskp, -1)

            cutimg = apply_mask(cutimg, cutmask, "black")

            fig, (ax1, ax2) = plt.subplots(
                1, 2, facecolor="none" if transparent_fc else "w", figsize=figsize
            )
            # Create a figure and axes, setting the facecolor to "none" (transparent)
            fig.patch.set_alpha(0)  # Transparency for the figure
            ax1.axis(False)
            ax2.axis(False)
            ax1.imshow(cropimg, alpha=1)
            ax2.imshow(cutimg, alpha=1)

            if title:
                species = row["common_name"]
                fontsize = (
                    fig.get_figwidth() + fig.get_figheight()
                ) * 0.5  # Adjust the scaling factor as desired

                # Add the main title
                fig.tight_layout()
                ax2.set_title(species, fontsize=fontsize)

            fig.tight_layout()
            new_save_location = Path(save_location, row["common_name"])
            new_save_location.mkdir(exist_ok=True, parents=True)
            cutout_stem = f"{row['cutout_id']}" + "_cutout_plot"

            plot_path = Path(
                new_save_location,
                cutout_stem + ".png" if transparent_fc else cutout_stem + ".jpg",
            )

            plt.savefig(
                plot_path, bbox_inches="tight", transparent=transparent_fc, dpi=dpi
            )
            plt.close()
