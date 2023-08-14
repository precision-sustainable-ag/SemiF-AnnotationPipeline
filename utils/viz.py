import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from tqdm import tqdm

sys.path.append("/home/psa_images/SemiF-AnnotationPipeline")
sys.path.append("/home/psa_images/SemiF-AnnotationPipeline/segment")
from segment.semif_utils.utils import apply_mask, read_json


def validation_sample_df(df, sample_sz=10, random_state=42, drop_img_duplicates=True):
    if drop_img_duplicates:
        df = df.drop_duplicates(
            subset="image_paths", keep="first"
        )  # Emphasize different full-res images and not cutouts
    df = df.sample(sample_sz, random_state=random_state)
    return df


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


def review_data(df, batch_id):
    print("\nBatch: ", batch_id)
    print("\nTotal number of cutouts by species")
    print(df.groupby(["common_name"])["cutout_id"].nunique())
    print(len(df))


def read_metadata(path):
    with open(path, "r") as f:
        data = json.loads(f.read())
    return data


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


def species_info2color_map(species_info):
    data = read_json(species_info)
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


# def inspect_masks(df, sample_size, figsize=(8,12),random_state=42):


def filter_by_area(df, area_min, area_max):
    desc = df["area"].describe()
    if area_min == "mean":
        df = df[df["area"] > desc.iloc[1]]
    if area_max == "mean":
        df = df[df["area"] < desc.iloc[1]]
    if area_min == 25:
        df = df[df["area"] > desc.iloc[4]]
    if area_max == 25:
        df = df[df["area"] < desc.iloc[4]]
    if area_min == 50:
        df = df[df["area"] > desc.iloc[5]]
    if area_max == 50:
        df = df[df["area"] < desc.iloc[5]]
    if area_min == 75:
        df = df[df["area"] > desc.iloc[6]]
    if area_max == 75:
        df = df[df["area"] < desc.iloc[6]]
    return df


def filter_by_solidity(df, solid_min, solid_max):
    if solid_min != None:
        df = df[df["solidity"] > solid_min]

    if solid_max != None:
        df = df[df["solidity"] < solid_max]

    return df


def filter_by_num_components(df, component_min, component_max):
    if component_min != None:
        df = df[df["num_components"] > component_min]

    if component_max != None:
        df = df[df["num_components"] < component_max]

    return df


def plot_bboxes(
    df,
    show_labels=True,
    save=False,
    transparent_fc=True,
    save_location=".",
    axis=False,
    figsize=(8, 12),
    show_plots=True,
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
        cv2.imread(image_path)
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
            image = plt.imread(image_path)
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

            if save:
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

                print("Saved image with detections to %s" % plot_path)
            if show_plots:
                plt.show()
            plt.close()


def inspect_masks(
    df_sample,
    figsize=(8, 12),
    transparent_fc=True,
    include_suptitles=True,
    save=False,
    save_location=".",
    colorize_semantic_mask=False,
    species_info=None,
    show_plots=True,
    dpi=300,
):
    # dfs = df.sample(sample_size, random_state=random_state)
    for _, df in df_sample.iterrows():
        print("Common name: ", df.common_name)
        imgpath = df["image_paths"]
        maskpath = df["semantic_masks"]
        instancepath = df["instance_masks"]
        bgr = cv2.imread(imgpath)
        rgbimg = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        bgrsemmask = cv2.imread(maskpath)
        print("Unique semantic mask values (class_id): ", np.unique(bgrsemmask[..., 0]))

        rgbmask = cv2.cvtColor(bgrsemmask, cv2.COLOR_BGR2RGB)

        if colorize_semantic_mask:
            color_map = species_info2color_map(species_info)
            rgbmask = convert_mask_values(rgbmask, color_map)

        bgrinmask = cv2.imread(instancepath)
        rgbinstance = cv2.cvtColor(bgrinmask, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=figsize, facecolor="none" if transparent_fc else "w"
        )
        fontsize = (
            fig.get_figwidth() + fig.get_figheight()
        ) * 0.5  # Adjust the scaling factor as desired

        ax1.imshow(rgbimg)
        ax2.imshow(rgbmask)
        ax3.imshow(rgbinstance)

        ax1.axis(False)
        ax2.axis(False)
        ax3.axis(False)

        if include_suptitles:
            ax1.title.set_text("Color image")
            ax1.title.set_fontsize(fontsize)
            ax2.title.set_text("Semantic mask")
            ax2.title.set_fontsize(fontsize)
            ax3.title.set_text("Instance mask")
            ax3.title.set_fontsize(fontsize)

        plt.tight_layout()
        if save:
            save_location = Path(save_location)
            save_location.mkdir(exist_ok=True, parents=True)
            image_stem = Path(imgpath).stem + "_mask_plot"
            plot_path = Path(
                save_location,
                image_stem + ".png" if transparent_fc else image_stem + ".jpg",
            )
            fig.savefig(
                plot_path, bbox_inches="tight", transparent=transparent_fc, dpi=dpi
            )
            print("Saved image with detections to %s" % plot_path)
        if show_plots:
            plt.show()
        plt.close()


def preview_cutout_results(
    df,
    extends_border,
    is_primary,
    green_sum_max,
    green_sum_min,
    area_min,
    area_max,
    solid_min,
    solid_max,
    component_min,
    component_max,
    figsize=(8, 12),
    save=False,
    save_location=".",
    transparent_fc=True,
    title=True,
    show_plots=True,
    dpi=300,
):
    mdf = df.copy()
    if extends_border != None:
        mdf = mdf[mdf["extends_border"] == extends_border]
    if is_primary != None:
        mdf = mdf[mdf["is_primary"] == is_primary]
    if green_sum_min != None:
        mdf = mdf[mdf["green_sum"] > green_sum_min]
    if green_sum_max != None:
        mdf = mdf[mdf["green_sum"] < green_sum_max]
    if area_min != None and area_max != None:
        mdf = filter_by_area(mdf, area_min, area_max)
    if solid_min != None and solid_max != None:
        mdf = filter_by_solidity(mdf, solid_min, solid_max)
    if component_min != None or component_max != None:
        mdf = filter_by_num_components(mdf, component_min, component_max)

    print("\nNumber of filtered cutouts by species:")
    print(mdf.groupby(["common_name"])["cutout_id"].nunique())
    print(len(mdf))

    for species in mdf["common_name"].unique():
        sdf = mdf[mdf["common_name"] == species]

        if len(sdf) == 0:
            print(f"{species} None")
            continue

        print(species)

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
            if save:
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
            if show_plots:
                plt.show()
            plt.close()


def batch_species_count_plot(
    df,
    fig_save_dir=".",
    title=True,
    save=False,
    custom_palette=["#c75858", "#4a7a9d"],
    transparent=False,
):
    g = sns.catplot(
        data=df, y="common_name", kind="count", hue="is_primary", palette=custom_palette
    )
    # Iterate through the axes to add annotations
    for ax in g.axes.flat:
        # Get the patches (bars) and their heights
        for p in ax.patches:
            # Get the height of the bar (this is the count)
            width = p.get_width()
            # Add a text annotation for each bar, placing it just beside the bar
            ax.annotate(
                f"{int(width)}",
                (width, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
            )

    batch_id = df["batch_id"].iloc[0]
    if title:
        # Add a title
        g.fig.suptitle(f"{batch_id}", y=1.09)

    # Change the y-axis label
    g.set_axis_labels(y_var="Common Name")

    if save:
        save_cutout_path = Path(f"{fig_save_dir}/{batch_id}_cutouts_by_species.png")

        Path(save_cutout_path.parent).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            save_cutout_path, bbox_inches="tight", transparent=transparent, dpi=300
        )

    plt.show()


def compile_cutout_csvs(cutout_dir):
    # cutout_batch_csvs = cutout_dir.rglob("*.csv")
    # dfs = []
    # for x in cutout_batch_csvs:
    #     if not str(x.stem).startswith(".") and x.is_file():

    #         # csv = list(x.glob("*.csv"))[0]
    #         df = pd.read_csv(x)
    #         dfs.append(df)
    # df = pd.concat(dfs).reset_index(names="old_index")
    # return df
    cutout_batch_dirs = cutout_dir.glob("*")
    cutout_batch_dirs = [x for x in cutout_batch_dirs]
    dfs = []
    for x in tqdm(cutout_batch_dirs):
        if not str(x.stem).startswith(".") and x.is_dir():
            batch = x.stem
            csv_path = Path(cutout_dir, batch, batch + ".csv")
            if csv_path.is_file():
                df = pd.read_csv(csv_path, low_memory=False)
                dfs.append(df)
    df = pd.concat(dfs).reset_index(names="og_index")
    return df
