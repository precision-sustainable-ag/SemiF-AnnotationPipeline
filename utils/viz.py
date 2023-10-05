import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.legend_handler import HandlerTuple
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


def save_original_full_res_images(
    df, save_location=".", figsize=(8, 12), show_plots=False
):
    for _, i in df.iterrows():
        src_path = i["image_paths"]
        dst_dir = Path(save_location, "full_res_images")
        dst_dir.mkdir(exist_ok=True, parents=True)
        dst_path = Path(dst_dir, Path(src_path).name)
        shutil.copy2(src_path, dst_dir)
        if show_plots:
            img = cv2.cvtColor(cv2.imread(str(dst_path)), cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(img)
            ax.axis(False)
            plt.tight_layout()
            plt.show()
            plt.close()


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
        print(imgpath)
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
