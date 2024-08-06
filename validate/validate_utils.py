import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from tqdm import tqdm

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

def filter_unique_images_with_multiple_classes(df, column_a="image_id", column_b="class_id"):
    # Group by column_a and count unique values in column_b
    unique_counts = df.groupby(column_a)[column_b].nunique()
    
    # Filter to get the values of column_a that have more than one unique value in column_b
    filtered_a_values = unique_counts[unique_counts > 1].index
    
    # Filter the DataFrame to include only rows with the filtered column_a values
    filtered_df = df[df[column_a].isin(filtered_a_values)]
    
    return filtered_df

def validation_sample_df(df: pd.DataFrame, sample_sz=10, random_state=42, drop_img_duplicates=True):

    filtered_df = filter_unique_images_with_multiple_classes(df)
    species_dfs = []
    unique_classes = df["common_name"].unique()
    
    if sample_sz < len(unique_classes):
        class_sample_size = 1
    else:
        class_sample_size = sample_sz // len(unique_classes)

    for uniq_cls in unique_classes:
        class_df = df[df["common_name"]== uniq_cls]
        class_df = class_df.drop_duplicates(subset="image_id")

        if class_df.shape[0] < class_sample_size:
            class_sample_size = class_df.shape[0]

        df_species_sample = class_df.sample(class_sample_size, random_state=random_state)
        species_dfs.append(df_species_sample)
    
    
    sample_filtered_sz = 10

    filtered_dfs = []
    unique_filtered_classes = filtered_df["common_name"].unique()
    
    for uniq_filt_cls in unique_filtered_classes:
        filt_class_df = filtered_df[filtered_df["common_name"]== uniq_filt_cls]
        filt_class_df = filt_class_df.drop_duplicates(subset="image_id")
    
        if filt_class_df.shape[0] < sample_filtered_sz:
            sample_filtered_sz = filt_class_df.shape[0]

        filtered_df_species_sample = filt_class_df.sample(sample_filtered_sz, random_state=random_state)
        filtered_dfs.append(filtered_df_species_sample)


    df_sample = pd.concat(species_dfs + filtered_dfs)
    df_sample = df_sample.drop_duplicates(subset="image_id")
    df = df[df["image_paths"].isin(df_sample["image_paths"])]

    return df.sort_values(by="image_id")


def get_detection_data(jsonpath):
    meta = read_metadata(jsonpath)
    categories = meta["categories"]

    annotations = meta["annotations"]
    boxes = []
    labels = []
    for annotation in annotations:
        x1 = annotation["bbox_xywh"][0] # top left x
        y1 = annotation["bbox_xywh"][1] # top left y

        w = annotation["bbox_xywh"][2] # bbox width
        h = annotation["bbox_xywh"][3] # bbox height

        x2 = x1 + w
        y2 = y1 + h

        boxes.append([x1, y1, x2, y2])

        bbox_category_class_id = annotation["category_class_id"]
        bbox_category = None
        for cat in categories:
            if cat["class_id"] == bbox_category_class_id:
                bbox_category = cat
                break

        class_id = bbox_category["class_id"]
        common_name = bbox_category["common_name"]

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

def resize_bboxes(bboxes, original_width, original_height, new_width, new_height):
    """
    Resize a list of bounding boxes according to the new image dimensions.

    Parameters:
    - bboxes: A list of bounding boxes, where each bounding box is in (x, y, w, h) format.
    - original_width: The original width of the image.
    - original_height: The original height of the image.
    - new_width: The desired width of the resized image.
    - new_height: The desired height of the resized image.

    Returns:
    - resized_bboxes: A list of resized bounding boxes in (x, y, w, h) format.
    """
    # Calculate the scaling factors
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Resize the bounding boxes
    resized_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        resized_x = int(x * scale_x)
        resized_y = int(y * scale_y)
        resized_w = int(w * scale_x)
        resized_h = int(h * scale_y)
        resized_bboxes.append((resized_x, resized_y, resized_w, resized_h))

    return resized_bboxes

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
    unique_images_df = df.drop_duplicates(subset="image_paths")
    for _, row in tqdm(unique_images_df.iterrows(), total=unique_images_df.shape[0]):
        image_path = Path(row["image_paths"])
        assert image_path.exists()

        meta_path = str(image_path).replace(f"images/{image_path.name}", f"metadata/{image_path.stem}.json")
        assert Path(meta_path).exists()
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
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get the original dimensions of the image
            original_height, original_width = image.shape[:2]
            new_width = original_width//5
            new_height = original_height//5
            
            # Resize the image
            resized_image = cv2.resize(image, (new_width, new_height))

            resized_bboxes = resize_bboxes(bboxes, original_width, original_height, new_width, new_height)

            plt.imshow(resized_image)

            if not axis:
                ax.axis(False)

            # Iterate over all the bounding boxes
            linewidth = (fig.get_figwidth() + fig.get_figheight()) * 0.1
            fontsize = (
                fig.get_figwidth() + fig.get_figheight()
            ) * 0.5  # Adjust the scaling factor as desired
            for i, bbox in enumerate(resized_bboxes):
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
                        (ry - 40),
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

# Generate pastel colors
def generate_pastel_colors(n):
    # Generate n pastel colors by blending random colors with white
    base_colors = np.random.rand(n, 3)
    pastel_colors = (base_colors + 1.0) / 2.0  # Mix with white (add and divide by 2)
    return (pastel_colors * 255).astype(np.uint8)

def generate_bright_colors(n):
    # Generate n bright colors
    base_colors = np.random.rand(n, 3) * 0.8 + 0.2  # Ensure brightness by keeping colors above 0.2
    return (base_colors * 255).astype(np.uint8)

def plot_masks(
    df,
    figsize=(8, 12),
    transparent_fc=False,
    include_suptitles=True,
    save_location=".",
    species_info=None,
    dpi=150,
):
    unique_images_df = df.drop_duplicates(subset="image_paths")
    for _, row in tqdm(unique_images_df.iterrows(), total=unique_images_df.shape[0]):
        imgpath = row["image_paths"]
        maskpath = row["semantic_masks"]
        instancepath = row["instance_masks"]
        bgr = cv2.imread(imgpath)
        rgbimg = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        bgrsemmask = cv2.imread(maskpath)

        rgbmask = cv2.cvtColor(bgrsemmask, cv2.COLOR_BGR2RGB)

        color_map = species_info2color_map(species_info)
        rgbmask = convert_mask_values(rgbmask, color_map)

        mask = cv2.imread(instancepath, cv2.IMREAD_UNCHANGED)

        # Initialize an empty image for colored output
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Get unique instance values (excluding 0, which is the background)
        unique_values = np.unique(mask)
        unique_values = unique_values[unique_values != 0]

        # Assign pastel colors to each unique instance
        colors = generate_bright_colors(len(unique_values))

        # Create a lookup table to map unique values to colors
        lookup_table = np.zeros((np.max(unique_values) + 1, 3), dtype=np.uint8)
        lookup_table[unique_values] = colors

        # Apply the lookup table to the mask to create the colored mask
        colored_mask = lookup_table[mask]

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=figsize, facecolor="none" if transparent_fc else "w"
        )
        original_height, original_width = rgbimg.shape[:2]
        new_width = original_width//5
        new_height = original_height//5
        
        # Resize the image
        resized_rgbimg = cv2.resize(rgbimg, (new_width, new_height))
        resized_rgbmask = cv2.resize(rgbmask, (new_width, new_height))
        resized_rgbinstance = cv2.resize(colored_mask, (new_width, new_height))

        ax1.imshow(resized_rgbimg)
        ax2.imshow(resized_rgbmask)
        ax3.imshow(resized_rgbinstance)

        ax1.axis(False)
        ax2.axis(False)
        ax3.axis(False)

        if include_suptitles:
            imgdf = df[df["image_paths"]== imgpath]
            uniq_common_names = ", ".join(imgdf["common_name"].unique())
            uniq_meta_class_ids = ", ".join(imgdf["class_id"].unique().astype(str))
            uniq_mask_class_ids = ", ".join([str(x) for x in np.unique(bgrsemmask[..., 0]) if x != 0])
            
            fontsize = (
                fig.get_figwidth() + fig.get_figheight()
            ) * 0.5  # Adjust the scaling factor as desired
            ax1.title.set_text(f"Unique Common names: {uniq_common_names}")
            ax1.title.set_fontsize(fontsize)
            ax2.title.set_text(f"Unique mask class_ids: \n{uniq_mask_class_ids}")
            ax2.title.set_fontsize(fontsize)
            ax3.title.set_text(f"Unique metadata class_ids: \n{uniq_meta_class_ids}")
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
    unique_images_df = df.drop_duplicates(subset="image_paths")
    unique_cnames = unique_images_df["common_name"].unique()
    for species in tqdm(unique_cnames, total=unique_cnames.shape[0]):
        sdf = unique_images_df[unique_images_df["common_name"] == species]

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
