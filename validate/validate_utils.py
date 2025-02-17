import json
import logging
import shutil
import sys
from pathlib import Path
import seaborn as sns
import matplotlib
from matplotlib.legend_handler import HandlerTuple

import re
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


def compile_cutout_csvs(cutout_dir):
    """Globs cutout dir csvs from main cutout dir, creates dataframes
    for each one, then concatenates them all.

    Args:
        cutout_dir (_type_): _description_
    """
    data = Path(cutout_dir).glob("*")
    csvs = []
    for a in tqdm(data):
        csv = list(a.glob("*.csv"))
        if len(csv) > 0:
            csvs.append(csv[0])
    df = pd.concat([pd.read_csv(x, low_memory=False) for x in csvs])

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

def filter_unique_images_with_multiple_classes(df, column_a="image_id", column_b="category_class_id"):
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
    unique_classes = df["category_common_name"].unique()
    
    if sample_sz < len(unique_classes):
        class_sample_size = 1
    else:
        class_sample_size = sample_sz // len(unique_classes)

    for uniq_cls in unique_classes:
        class_df = df[df["category_common_name"]== uniq_cls]
        class_df = class_df.drop_duplicates(subset="image_id")

        if class_df.shape[0] < class_sample_size:
            class_sample_size = class_df.shape[0]

        df_species_sample = class_df.sample(class_sample_size, random_state=random_state)
        species_dfs.append(df_species_sample)
    
    
    sample_filtered_sz = 10

    filtered_dfs = []
    unique_filtered_classes = filtered_df["category_common_name"].unique()
    
    for uniq_filt_cls in unique_filtered_classes:
        filt_class_df = filtered_df[filtered_df["category_common_name"]== uniq_filt_cls]
        filt_class_df = filt_class_df.drop_duplicates(subset="image_id")
    
        if filt_class_df.shape[0] < sample_filtered_sz:
            sample_filtered_sz = filt_class_df.shape[0]

        filtered_df_species_sample = filt_class_df.sample(sample_filtered_sz, random_state=random_state)
        filtered_dfs.append(filtered_df_species_sample)


    df_sample = pd.concat(species_dfs + filtered_dfs)
    df_sample = df_sample.drop_duplicates(subset="image_id")
    df = df[df["image_paths"].isin(df_sample["image_paths"])]

    return df.sort_values(by="image_id")



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


def get_detection_data(jsonpath, simple_labels=False):
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

        label = f"{common_name} ({class_id})" if not simple_labels else f"{class_id}"
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

def get_bboxes_validation_images(
    df,
    processed_images,
    show_labels=True,
    resize_factor=0.2,  # Factor to resize the image
):
    """Draws bounding boxes directly on images using OpenCV.

    Args:
        df (pandas dataframe): DataFrame containing image paths and bounding boxes.
        processed_images (set): Set of image IDs that have already been reviewed.
        show_labels (bool, optional): Whether to display labels. Defaults to True.
        transparent_fc (bool, optional): If True, uses transparent face color. Defaults to True.
        resize_factor (float, optional): Factor by which the image should be resized. Defaults to 0.2.

    Returns:
        list: List of images with bounding boxes drawn.
    """
    annotated_images = []
    unique_images_df = df.drop_duplicates(subset="image_paths").sort_values("image_id")

    for _, row in tqdm(unique_images_df.iterrows(), total=unique_images_df.shape[0]):
        if row["image_id"] in processed_images and "bbox" in row["product"]:
            print(f"Skipping {row['image_id']} as it has already been reviewed.")
            continue
        image_path = Path(row["image_paths"])
        assert image_path.exists(), f"Image path does not exist: {image_path}"

        meta_path = str(image_path).replace(f"images/{image_path.name}", f"metadata/{image_path.stem}.json")
        assert Path(meta_path).exists(), f"Metadata path does not exist: {meta_path}"

        # Load bounding boxes and labels
        bboxes, labels = get_detection_data(meta_path)
        if len(bboxes) == 0:
            continue

        # Read image using OpenCV
        image = cv2.imread(str(image_path))
        original_height, original_width = image.shape[:2]

        # Resize image while maintaining aspect ratio
        new_width = int(original_width * resize_factor)
        new_height = int(original_height * resize_factor)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Resize bounding boxes to match resized image
        resized_bboxes = resize_bboxes(bboxes, original_width, original_height, new_width, new_height)

        # Draw bounding boxes on image
        for i, bbox in enumerate(resized_bboxes):
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            # Draw rectangle (bounding box)
            cv2.rectangle(resized_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color

            if show_labels and labels is not None:
                label_text = str(labels[i])
                font_scale = 0.6  # Scale label text dynamically
                thickness = 2
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                text_x = xmin
                text_y = max(ymin - 10, 10)  # Position label above the bbox

                # Draw background rectangle for text (optional, improves readability)
                cv2.rectangle(
                    resized_image,
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 255),  # Red background
                    -1  # Filled rectangle
                )

                # Put text label on the bounding box
                cv2.putText(
                    resized_image, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
                )

        # Append processed image to the list
        annotated_images.append((resized_image, row))

    return annotated_images

def get_mask_validation_images(
    df,
    processed_images,
    species_info=None,
    resize_factor=0.2,  # Factor to resize images and masks
    include_suptitles=True
):
    """Generates images with masks and bounding boxes using OpenCV.

    Args:
        df (pandas dataframe): DataFrame containing image paths and mask paths.
        processed_images (set): Set of image IDs that have already been reviewed.
        species_info (dict, optional): Species info for mask color mapping.
        resize_factor (float, optional): Factor by which the image should be resized. Defaults to 0.2.
        include_suptitles (bool, optional): Whether to overlay text info on images.

    Returns:
        list: List of annotated images.
    """
    annotated_images = []
    unique_images_df = df.drop_duplicates(subset="image_paths")
    
    for _, row in tqdm(unique_images_df.iterrows(), total=unique_images_df.shape[0]):
        if row["image_id"] in processed_images and "mask" in row["product"]:
            print(f"Skipping {row['image_id']} as it has already been reviewed.")
            continue
        imgpath = Path(row["image_paths"])
        maskpath = row["semantic_masks"]
        assert imgpath.exists(), f"Image path does not exist: {imgpath}"
        assert Path(maskpath).exists(), f"Mask path does not exist: {maskpath}"

        meta_path = str(imgpath).replace(f"images/{imgpath.name}", f"metadata/{imgpath.stem}.json")
        bboxes, labels = get_detection_data(meta_path, simple_labels=True)

        # Read images and masks
        image = cv2.imread(str(imgpath))
        mask = cv2.imread(str(maskpath))

        # Convert BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Apply species-specific color map to mask
        color_map = species_info2color_map(species_info)
        mask = convert_mask_values(mask, color_map)

        original_height, original_width = image.shape[:2]

        # Resize images & masks while keeping proportions
        new_width = int(original_width * resize_factor)
        new_height = int(original_height * resize_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.resize(mask, (new_width, new_height))
        resized_bboxes = resize_bboxes(bboxes, original_width, original_height, new_width, new_height)

        # Create a side-by-side visualization (original, mask, mask+bboxes)
        combined_image = np.hstack([resized_image, resized_mask, resized_mask.copy()])

        # Overlay bounding boxes on the third section (mask with bboxes)
        for i, bbox in enumerate(resized_bboxes):
            xmin, ymin, xmax, ymax = map(int, bbox)

            # Draw bounding box in red
            cv2.rectangle(combined_image[:, new_width*2:], (xmin, ymin), (xmax, ymax), (0, 0, 255), 8)

            # Label the bounding box
            if labels is not None:
                label_text = str(labels[i])
                font_scale = 1.5
                thickness = 2
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                text_x = xmin
                text_y = max(ymin - 10, 10)  # Place label above bbox

                # Draw background for text
                cv2.rectangle(
                    combined_image[:, new_width*2:],
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 255),  # Red background
                    -1  # Filled rectangle
                )

                # Overlay text
                cv2.putText(
                    combined_image[:, new_width*2:], label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
                )

        # Overlay textual metadata on the first image if enabled
        if include_suptitles:
            imgdf = df[df["image_paths"] == str(imgpath)]
            uniq_common_names = ", ".join(imgdf["category_common_name"].unique())
            uniq_meta_class_ids = ", ".join(imgdf["category_class_id"].unique().astype(str))
            uniq_mask_class_ids = ", ".join([str(x) for x in np.unique(mask[..., 0]) if x != 0])

            metadata_texts = [
                f"Common Names: {uniq_common_names}",
                f"Mask Class IDs: {uniq_mask_class_ids}",
                f"Metadata Class IDs: {uniq_meta_class_ids}"
            ]

            for idx, text in enumerate(metadata_texts):
                cv2.putText(
                    combined_image, text, (10, 30 + (idx * 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

        # Append processed image to the list
        annotated_images.append((combined_image, row))

    return annotated_images

def get_cutout_validate_images(
    df,
    processed_images,
    resize_factor=0.2,  # Factor to resize images
    title=True
):
    """Generates cutout visualizations using OpenCV.

    Args:
        df (pandas dataframe): DataFrame containing image paths and cutouts.
        resize_factor (float, optional): Factor by which the image should be resized. Defaults to 0.2.
        title (bool, optional): Whether to overlay species name.

    Returns:
        list: List of annotated images.
    """
    annotated_images = []
    unique_images_df = df.drop_duplicates(subset="image_paths")

    unique_cnames = unique_images_df["category_common_name"].unique()
    
    for species in tqdm(unique_cnames, total=unique_cnames.shape[0]):
        sdf = unique_images_df[unique_images_df["category_common_name"] == species]

        if len(sdf) == 0:
            continue

        for _, row in sdf.iterrows():
            if row["image_id"] in processed_images and "cutout" in row["product"]:
                print(f"Skipping {row['image_id']} as it has already been reviewed.")
                continue
            cutimgp = row["cutout_paths"]
            cropimgp = row["cutout_paths"].replace(".png", ".jpg")
            cutmaskp = row["cutout_paths"].replace(".png", "_mask.png")

            assert Path(cutimgp).exists(), f"Cutout image path does not exist: {cutimgp}"
            assert Path(cropimgp).exists(), f"Crop image path does not exist: {cropimgp}"
            assert Path(cutmaskp).exists(), f"Mask image path does not exist: {cutmaskp}"

            # Load images and mask
            crop_img = cv2.imread(cropimgp)
            cut_img = cv2.imread(cutimgp)
            cut_mask = cv2.imread(cutmaskp, cv2.IMREAD_GRAYSCALE)

            # Convert to RGB
            # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # cut_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2RGB)

            # Apply mask to cutout
            cut_img = apply_mask(cut_img, cut_mask, "black")

            # Resize images while keeping proportions
            original_height, original_width = crop_img.shape[:2]
            original_area = original_height * original_width
            
            if original_area < 10000:
                resize_factor = 2.0
                
            
            new_width = int(original_width * resize_factor)
            new_height = int(original_height * resize_factor)

            resized_crop_img = cv2.resize(crop_img, (new_width, new_height))

            resized_cut_img = cv2.resize(cut_img, (new_width, new_height))
    
            
            # Create side-by-side visualization
            combined_image = np.hstack([resized_crop_img, resized_cut_img])

            # Overlay species name as a title
            if title:
                label_text = row["category_common_name"]
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                text_x = 10
                text_y = 30

                # Draw background rectangle for text
                cv2.rectangle(
                    combined_image,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),  # Black background
                    -1  # Filled rectangle
                )

                # Put text label on the image
                cv2.putText(
                    combined_image, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
                )

            # Append processed image to the list
            annotated_images.append((combined_image, row))

    return annotated_images


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
        imgpath = Path(row["image_paths"])
        maskpath = row["semantic_masks"]
        meta_path = str(imgpath).replace(f"images/{imgpath.name}", f"metadata/{imgpath.stem}.json")
        bboxes, labels = get_detection_data(meta_path)
        
        # instancepath = row["instance_masks"]
        bgr = cv2.imread(str(imgpath))
        rgbimg = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        bgrsemmask = cv2.imread(maskpath)

        rgbmask = cv2.cvtColor(bgrsemmask, cv2.COLOR_BGR2RGB)

        color_map = species_info2color_map(species_info)
        rgbmask = convert_mask_values(rgbmask, color_map)

        # instance_mask = cv2.imread(instancepath, cv2.IMREAD_UNCHANGED)

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=figsize, facecolor="none" if transparent_fc else "w"
        )
        original_height, original_width = rgbimg.shape[:2]
        new_width = original_width//5
        new_height = original_height//5
        
        # Resize the image
        resized_rgbimg = cv2.resize(rgbimg, (new_width, new_height))
        resized_rgbmask = cv2.resize(rgbmask, (new_width, new_height))
        resized_bboxes = resize_bboxes(bboxes, original_width, original_height, new_width, new_height)
        # resized_rgbinstance = cv2.resize(instance_mask, (new_width, new_height))

        fontsize = (
            fig.get_figwidth() + fig.get_figheight()
        ) * 0.5  # Adjust the scaling factor as desired
        
        ax1.imshow(resized_rgbimg)
        ax2.imshow(resized_rgbmask)
        ax3.imshow(resized_rgbmask)

        ax1.axis(False)
        ax2.axis(False)
        ax3.axis(False)
        for i, bbox in enumerate(resized_bboxes):
                xmin, ymin, xmax, ymax = bbox
                w = xmax - xmin
                h = ymax - ymin

                # add bounding boxes to the image
                # Make line width dynamic based on bbox size
                box = patches.Rectangle(
                    (xmin, ymin),
                    w,
                    h,
                    linewidth=0.25,
                    edgecolor="red",
                    facecolor="none",
                )

                ax3.add_patch(box)

        if include_suptitles:
            imgdf = df[df["image_paths"]== imgpath]
            uniq_common_names = ", ".join(imgdf["category_common_name"].unique())
            uniq_meta_class_ids = ", ".join(imgdf["category_class_id"].unique().astype(str))
            uniq_mask_class_ids = ", ".join([str(x) for x in np.unique(bgrsemmask[..., 0]) if x != 0])
            
            fontsize = (
                fig.get_figwidth() + fig.get_figheight()
            ) * 0.5  # Adjust the scaling factor as desired
            ax1.title.set_text(f"Unique Common names: {uniq_common_names}")
            ax1.title.set_fontsize(fontsize)
            ax2.title.set_text(f"Unique mask class_ids: \n{uniq_mask_class_ids}")
            ax2.title.set_fontsize(fontsize)
            ax3.title.set_text(f"Unique metadata class_ids: \n{uniq_meta_class_ids}")
            # ax3.title.set_fontsize(fontsize)

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
    unique_cnames = unique_images_df["category_common_name"].unique()
    for species in tqdm(unique_cnames, total=unique_cnames.shape[0]):
        sdf = unique_images_df[unique_images_df["category_common_name"] == species]

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
                species = row["category_common_name"]
                fontsize = (
                    fig.get_figwidth() + fig.get_figheight()
                ) * 0.5  # Adjust the scaling factor as desired

                # Add the main title
                fig.tight_layout()
                ax2.set_title(species, fontsize=fontsize)

            fig.tight_layout()
            new_save_location = Path(save_location, row["category_common_name"])
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


def plot_images_by_state(df, hex_color, plot_title, save=False):
    sns.set_style("dark")
    sns.set_context("notebook")
    col_strings = df.season.unique()
    col_order = sorted(col_strings, key=custom_sort)
    g = sns.catplot(
        data=df,
        x="common_name",
        y="count",
        col="season",
        kind="bar",
        # hue="is_primary",
        sharex=False,
        errorbar=None,
        color=hex_color,
        col_order=col_order,
    )

    # Replace underscores between numbers with '/'
    modified_strings = [re.sub(r"(\d)_(\d)", r"\1/\2", s) for s in col_order]

    # Replace remaining underscores with spaces
    new_titles = [s.replace("_", " ") for s in modified_strings]
    for i, ax in enumerate(g.axes.flat):
        # Annotate the bars with their heights (the 'count' values)
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}" if not pd.isna(p.get_height()) else 0.0,
                xy=(p.get_x() + p.get_width() / 2.0, p.get_height()),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            ax.set_title(new_titles[i])
    g.set_xticklabels(rotation=75)
    # Change the y-axis label
    g.set_axis_labels(y_var="")
    g.set_axis_labels(x_var="")
    g.set_yticklabels(None)
    g.fig.suptitle(plot_title, y=1.13, fontsize=18)
    if save:
        g.savefig(f"plots/{plot_title}.png", dpi=300)
    plt.show()


def plot_sub_images(df, binary_palettes, title, save=False):
    sns.set_style("dark")
    sns.set_context("notebook")
    col_strings = df.season.unique()
    col_order = sorted(col_strings, key=custom_sort)
    g = sns.catplot(
        data=df,
        x="common_name",
        y="count",
        col="season",
        kind="bar",
        hue="is_primary",
        sharex=False,
        errorbar=None,
        palette=binary_palettes,
        col_order=col_order,
    )

    # Replace underscores between numbers with '/'
    modified_strings = [re.sub(r"(\d)_(\d)", r"\1/\2", s) for s in col_order]

    # Replace remaining underscores with spaces
    new_titles = [s.replace("_", " ") for s in modified_strings]

    for i, ax in enumerate(g.axes.flat):
        # Annotate the bars with their heights (the 'count' values)
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}" if not pd.isna(p.get_height()) else 0.0,
                xy=(p.get_x() + p.get_width() / 2.0, p.get_height()),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            ax.set_title(new_titles[i])
        # ax.axis(False)
    g.set_xticklabels(rotation=75)
    # Change the y-axis label
    g.set_axis_labels(y_var="")
    g.set_axis_labels(x_var="")
    g.set_yticklabels(None)

    g.fig.suptitle(title, y=1.13, fontsize=18)
    if save:
        g.savefig(f"plots/{title}.png", dpi=300)
    plt.show()


def read_container_list_summary(path):
    # Read the file content
    with open(path, "r") as file:
        your_text_file_content = file.read()

    # Create a DataFrame to hold the extracted data
    data = []

    # Extract and structure the data
    lines = your_text_file_content.split("\n")  # Split the text file into lines
    state = None
    for line in lines:
        if "State:" in line:
            state = line.split(":")[1].strip()
        elif "Category:" in line:
            category = line.split("-")[0].replace("Category:", "").strip()
            processed = int(line.split("Processed:")[1].split(",")[0].strip())
            not_processed = int(line.split("Not Processed:")[1].strip())
            data.append([state, category, "Processed", processed])
            data.append([state, category, "Not Processed", not_processed])
    dftemp = pd.DataFrame(
        data, columns=["State", "Category", "Processed status", "total"]
    ).sort_values(by=["State", "Category"])
    return dftemp


def plot_processed_batches(df, plot_title="plot", save=False, height=6, aspect=0.7):
    sns.set(style="dark", context="notebook", font_scale=1.4)
    # Sort strings based on the average year key
    col_strings = df.Category.unique()
    col_order = sorted(col_strings, key=custom_sort)

    g = sns.catplot(
        x="State",
        y="total",
        hue="Processed status",
        hue_order=df["Processed status"].unique()[::-1],
        col="Category",
        col_order=col_order,
        data=df,
        kind="bar",
        height=height,
        aspect=aspect,
        errorbar=None,
        legend=False,
    )
    dark_colors = ["#1565C0", "#388E3C", "#C62828"]
    light_colors = ["#90CAF9", "#A5D6A7", "#FFCDD2"]

    # Replace underscores between numbers with '/'
    modified_strings = [re.sub(r"(\d)_(\d)", r"\1/\2", s) for s in col_order]

    # Replace remaining underscores with spaces
    new_titles = [s.replace("_", " ") for s in modified_strings]

    for h, ax in enumerate(g.axes.flat):
        col_value = ax.get_title().split(" = ")[-1]
        xticks = [tick.get_text() for tick in ax.get_xticklabels()]
        for i, bar_group in enumerate(ax.containers):
            for j, bar in enumerate(bar_group):
                category = xticks[j % len(xticks)]
                filtered_df = df[
                    (df["State"] == category)
                    & (
                        np.isclose(df["total"], bar.get_height(), atol=1e-2)
                    )  # I added a small tolerance here, but you can adjust it
                ]
                if not filtered_df.empty:
                    hue_value = filtered_df["Processed status"].iloc[0]
                    if hue_value == "Not Processed":
                        bar.set_facecolor(light_colors[j])
                    else:
                        bar.set_facecolor(dark_colors[j])

                    # Annotate each bar with its height (the 'count' value)
                    height_from_bar = (
                        int(bar.get_height()) if not pd.isna(bar.get_height()) else 0
                    )
                    height_from_df = int(filtered_df["total"].iloc[0])
                    ax.annotate(
                        str(height_from_bar),
                        xy=(bar.get_x() + bar.get_width() / 2.0, height_from_bar),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                    # Verify and print discrepancies
                    if height_from_bar != height_from_df:
                        print(f"Discrepancy detected: Bar {j} in Group {i} in Axis {h}")
                        print(
                            f"Bar value: {height_from_bar} vs. DataFrame value: {height_from_df}"
                        )
        ax.set_title(new_titles[h])

    # Create a list of custom colors
    custom_colors = light_colors + dark_colors
    # Create custom legend handles
    legend_handles = [tuple(bar_group) for bar_group in ax.containers]

    # Create a list of legend labels
    legend_labels = [bar_group.get_label() for bar_group in ax.containers]
    counter = 0
    for i, handle in enumerate(legend_handles):
        for j in handle:
            j.set_facecolor(custom_colors[counter])
            counter += 1
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="Status",
        handlelength=4,
        fontsize=12,
        title_fontsize=16,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
    )

    g.set_axis_labels(y_var="")
    g.set_axis_labels(x_var="")
    g.set_yticklabels(None)
    sns.despine()
    plt.suptitle(plot_title, fontsize=18)
    plt.tight_layout()
    if save:
        g.savefig(f"plots/{plot_title}.png", dpi=300)
    plt.show()


def plot_images_by_location(df, title="Images by season", save=True):
    sns.set_style("dark")
    sns.set_context("notebook")
    custom_palette = ["#7BACD6", "#82C09A", "#FF8383"]
    # Sort strings based on the average year key
    col_strings = df.season.unique()
    col_order = sorted(col_strings, key=custom_sort)
    g = sns.catplot(
        data=df,
        x="state_id",
        kind="count",
        col="season",
        palette=custom_palette,
        col_order=col_order,
        # sharex=False,
    )
    # Replace underscores between numbers with '/'
    modified_strings = [re.sub(r"(\d)_(\d)", r"\1/\2", s) for s in col_order]

    # Replace remaining underscores with spaces
    new_titles = [s.replace("_", " ") for s in modified_strings]
    # Iterate through the axes to add annotations
    for i, ax in enumerate(g.axes.flat):
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}" if not pd.isna(p.get_height()) else 0.0,
                xy=(p.get_x() + p.get_width() / 2.0, p.get_height()),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            ax.set_title(new_titles[i])

    # Change the y-axis label
    g.set_axis_labels(y_var="")
    g.set_axis_labels(x_var="")
    g.set_yticklabels(None)

    g.fig.suptitle(title, y=1.13, fontsize=18)
    if save:
        g.savefig(f"plots/{title}.png", dpi=300)
    plt.show()


def custom_sort(s):
    # Extract years. If there are two years, take the first one for sorting
    year = int(re.findall(r"\d+", s)[0])

    # Add a sort key to ensure 'cash_crops' comes before 'weeds'
    if "cash_crops" in s.lower() or "cash crops" in s.lower():
        crop_type = 0
    elif "weeds" in s.lower():
        crop_type = 1
    else:
        crop_type = 2

    return (year, crop_type)


def plot_cutouts_by_state_and_season_all(df, plot_title, save=False):
    sns.set(style="dark", context="notebook", font_scale=1.4)
    # Sort strings based on the average year key
    col_strings = df.season.unique()
    col_order = sorted(col_strings, key=custom_sort)
    g = sns.catplot(
        x="state_id",
        y="count",
        # hue="is_primary",
        kind="bar",
        col="season",
        # ax=ax,
        col_order=col_order,
        data=df,
        errorbar=None,
        legend=False,
    )

    dark_colors = ["#1565C0", "#388E3C", "#C62828"]
    light_colors = ["#90CAF9", "#A5D6A7", "#FFCDD2"]

    # Replace underscores between numbers with '/'
    modified_strings = [re.sub(r"(\d)_(\d)", r"\1/\2", s) for s in col_order]

    # Replace remaining underscores with spaces
    new_titles = [s.replace("_", " ") for s in modified_strings]

    for h, ax in enumerate(g.axes.flat):
        col_value = ax.get_title().split(" = ")[-1]
        xticks = [tick.get_text() for tick in ax.get_xticklabels()]
        for i, bar_group in enumerate(ax.containers):
            for j, bar in enumerate(bar_group):
                category = xticks[j % len(xticks)]
                filtered_df = df[
                    (df["state_id"] == category)
                    & (
                        np.isclose(df["count"], bar.get_height(), atol=1e-2)
                    )  # I added a small tolerance here, but you can adjust it
                ]

                if not filtered_df.empty:
                    hue_value = filtered_df["is_primary"].iloc[0]
                    if hue_value == 0:
                        bar.set_facecolor(light_colors[j])
                    else:
                        bar.set_facecolor(dark_colors[j])

                    # Annotate each bar with its height (the 'count' value)
                    height_from_bar = (
                        int(bar.get_height()) if not pd.isna(bar.get_height()) else 0
                    )
                    height_from_df = int(filtered_df["count"].iloc[0])
                    ax.annotate(
                        str(height_from_bar),
                        xy=(bar.get_x() + bar.get_width() / 2.0, height_from_bar),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                    # Verify and print discrepancies
                    if height_from_bar != height_from_df:
                        print(f"Discrepancy detected: Bar {j} in Group {i} in Axis {h}")
                        print(
                            f"Bar value: {height_from_bar} vs. DataFrame value: {height_from_df}"
                        )
        ax.set_title(new_titles[h])

    # Create a list of custom colors
    custom_colors = light_colors + dark_colors
    # Create custom legend handles
    legend_handles = [tuple(bar_group) for bar_group in ax.containers]

    # Create a list of legend labels
    legend_labels = [bar_group.get_label() for bar_group in ax.containers]
    counter = 0
    for i, handle in enumerate(legend_handles):
        for j in handle:
            j.set_facecolor(custom_colors[counter])
            counter += 1
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="Unique cutouts",
        handlelength=4,
        fontsize=12,
        title_fontsize=16,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
    )

    g.set_axis_labels(y_var="")
    g.set_axis_labels(x_var="")
    g.set_yticklabels(None)
    sns.despine()
    plt.suptitle(plot_title, fontsize=18)
    plt.tight_layout()
    if save:
        g.savefig(f"plots/{plot_title}.png", dpi=300)
    plt.show()


def plot_cutouts_by_state_and_season(df, plot_title, save=False):
    sns.set(style="dark", context="notebook", font_scale=1.4)
    # Sort strings based on the average year key
    col_strings = df.season.unique()
    col_order = sorted(col_strings, key=custom_sort)
    g = sns.catplot(
        x="state_id",
        y="count",
        hue="is_primary",
        kind="bar",
        col="season",
        # ax=ax,
        col_order=col_order,
        data=df,
        errorbar=None,
        legend=False,
    )

    dark_colors = ["#1565C0", "#388E3C", "#C62828"]
    light_colors = ["#90CAF9", "#A5D6A7", "#FFCDD2"]

    # Replace underscores between numbers with '/'
    modified_strings = [re.sub(r"(\d)_(\d)", r"\1/\2", s) for s in col_order]

    # Replace remaining underscores with spaces
    new_titles = [s.replace("_", " ") for s in modified_strings]

    for h, ax in enumerate(g.axes.flat):
        col_value = ax.get_title().split(" = ")[-1]
        xticks = [tick.get_text() for tick in ax.get_xticklabels()]
        for i, bar_group in enumerate(ax.containers):
            for j, bar in enumerate(bar_group):
                category = xticks[j % len(xticks)]
                filtered_df = df[
                    (df["state_id"] == category)
                    & (
                        np.isclose(df["count"], bar.get_height(), atol=1e-2)
                    )  # I added a small tolerance here, but you can adjust it
                ]

                if not filtered_df.empty:
                    hue_value = filtered_df["is_primary"].iloc[0]
                    if hue_value == 0:
                        bar.set_facecolor(light_colors[j])
                    else:
                        bar.set_facecolor(dark_colors[j])

                    # Annotate each bar with its height (the 'count' value)
                    height_from_bar = (
                        int(bar.get_height()) if not pd.isna(bar.get_height()) else 0
                    )
                    height_from_df = int(filtered_df["count"].iloc[0])
                    ax.annotate(
                        str(height_from_bar),
                        xy=(bar.get_x() + bar.get_width() / 2.0, height_from_bar),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                    # Verify and print discrepancies
                    if height_from_bar != height_from_df:
                        print(f"Discrepancy detected: Bar {j} in Group {i} in Axis {h}")
                        print(
                            f"Bar value: {height_from_bar} vs. DataFrame value: {height_from_df}"
                        )
        ax.set_title(new_titles[h])

    # Create a list of custom colors
    custom_colors = light_colors + dark_colors
    # Create custom legend handles
    legend_handles = [tuple(bar_group) for bar_group in ax.containers]

    # Create a list of legend labels
    legend_labels = [bar_group.get_label() for bar_group in ax.containers]
    counter = 0
    for i, handle in enumerate(legend_handles):
        for j in handle:
            j.set_facecolor(custom_colors[counter])
            counter += 1
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="Unique cutouts",
        handlelength=4,
        fontsize=12,
        title_fontsize=16,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)},
    )

    g.set_axis_labels(y_var="")
    g.set_axis_labels(x_var="")
    g.set_yticklabels(None)
    sns.despine()
    plt.suptitle(plot_title, fontsize=18)
    plt.tight_layout()
    if save:
        g.savefig(f"plots/{plot_title}.png", dpi=300)
    plt.show()


def plot_sub_images_subplots(
    df,
    title,
    suptitle_fontsize=12,
    xtick_fontsize=12,
    bar_fontsize=9,
    hspace=0.4,
    wspace=0.1,
    save=False,
    plot_height=6,
    aspect=1.2,
):
    sns.set_style("dark")
    sns.set_context("notebook")
    col_strings = df.season.unique()
    col_order = sorted(col_strings, key=custom_sort)
    g = sns.catplot(
        data=df,
        x="common_name",
        y="count",
        col="season",
        kind="bar",
        hue="is_primary",
        row="state_id",
        height=plot_height,
        aspect=aspect,
        sharex=False,
        errorbar=None,
        legend=False,
        col_order=col_order,
    )

    # Replace underscores between numbers with '/'
    modified_strings = [re.sub(r"(\d)_(\d)", r"\1/\2", s) for s in col_order]
    # Replace remaining underscores with spaces
    hue_map = {
        "MD": ["#1565C0", "#90CAF9"],
        "NC": ["#388E3C", "#A5D6A7"],
        "TX": ["#C62828", "#FFCDD2"],
    }
    # Replace underscores between numbers with '/'
    modified_strings = [re.sub(r"(\d)_(\d)", r"\1/\2", s) for s in col_order]

    # Replace remaining underscores with spaces
    new_titles = [s.replace("_", " ") for s in modified_strings]
    for _, row_axes in enumerate(g.axes):  # Iterate over rows of the grid
        # Variables to store labels and colors for the current row
        current_labels = []
        current_colors = []

        for j, ax in enumerate(row_axes):  # Iterate over individual axes
            row_state_id = ax.get_title()
            pattern = (
                r"state_id\s*=\s*(?P<state_id>\w+)\s*\|\s*season\s*=\s*(?P<season>\w+)"
            )
            match = re.search(pattern, row_state_id)
            # Check if there's a match and if the state ID is in the dictionary keys
            if match:
                state_id = match.group("state_id")
                season = match.group("season")
                # Optionally check if state_id is in the dictionary keys
                if state_id not in hue_map:
                    state_id = None
            else:
                state_id, season = None, None
            # Check if there are bars in the subplot with positive height
            bars_count = sum(1 for bar in ax.patches if bar.get_height() > 0)
            if bars_count == 0:
                ax.set_xticks([])  # Removes x-ticks and their labels
            for bar_group in ax.containers:
                # Extracting hue value from legend labels
                hue_label = bar_group.get_label().lower() == "true"
                bar_hue = hue_map[state_id][0 if hue_label else 1]
                if hue_label not in current_labels:
                    current_labels.append(hue_label)
                    current_colors.append(bar_hue)
                for bar in bar_group:  # Iterate over individual bars in that group
                    bar.set_facecolor(bar_hue)
                    # Annotate each bar with its height (the 'count' value)
                    height_from_bar = (
                        int(bar.get_height()) if not pd.isna(bar.get_height()) else 0
                    )
                    ax.annotate(
                        str(height_from_bar),
                        xy=(bar.get_x() + bar.get_width() / 2.0, height_from_bar),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=bar_fontsize,
                    )
            ax.set_title(f"{state_id} | {new_titles[j]}", fontsize=suptitle_fontsize)
        handles = [matplotlib.patches.Patch(color=color) for color in current_colors]
        ax.legend(
            handles=handles,
            labels=current_labels,
            title="Is Primary?",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

    g.set_xticklabels(rotation=75, fontsize=xtick_fontsize)
    # # Change the y-axis label
    g.set_axis_labels(y_var="")
    g.set_axis_labels(x_var="")
    g.set_yticklabels(None)
    # Adjust spacing between plots
    g.fig.subplots_adjust(hspace=hspace, wspace=wspace)

    g.fig.suptitle(title, y=1.13, fontsize=18)
    if save:
        g.savefig(f"plots/{title}.png", dpi=300)
    plt.show()
