import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

def iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes in (xmin, ymin, xmax, ymax) format."""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # Calculate intersection coordinates
    xi1 = max(xmin1, xmin2)
    yi1 = max(ymin1, ymin2)
    xi2 = min(xmax1, xmax2)
    yi2 = min(ymax1, ymax2)

    # Calculate the area of the intersection rectangle
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the areas of both bounding boxes
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    # Calculate IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def is_contained(box1, box2):
    """Check if box2 is fully contained within box1."""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    return (xmin1 <= xmin2 <= xmax2 <= xmax1) and (ymin1 <= ymin2 <= ymax2 <= ymax1)

def merge_boxes(boxes, threshold=0.5):
    """Merge overlapping or fully contained bounding boxes into a single larger one."""
    merged_boxes = []

    while boxes:
        # Take the first box
        current_box = boxes.pop(0)
        xmin, ymin, xmax, ymax = current_box

        to_merge = [current_box]
        remaining_boxes = []

        # Check for overlap or containment with remaining boxes
        for box in boxes:
            if iou(current_box, box) >= threshold or is_contained(current_box, box) or is_contained(box, current_box):
                # Merge with overlapping or contained box by updating boundaries
                xmin = min(xmin, box[0])
                ymin = min(ymin, box[1])
                xmax = max(xmax, box[2])
                ymax = max(ymax, box[3])
                to_merge.append(box)
            else:
                remaining_boxes.append(box)

        # Add the new merged box to the result list
        new_box = [xmin, ymin, xmax, ymax]
        merged_boxes.append(new_box)

        # Continue with the remaining boxes
        boxes = remaining_boxes

    return merged_boxes


def process_csv_file(csv_path, iou_threshold=0.5):
    """Process a single CSV, merge target bounding boxes, and retain non-target rows."""
    # Read the CSV
    df = pd.read_csv(csv_path)

    if df.empty:
        log.warning(f"No detections found in: {csv_path}")
        return
    
    # Separate target and non-target rows
    target_df = df[df['classifier_classname'] == 'target_weed']
    nontarget_df = df[df['classifier_classname'] != 'target_weed']
    # drop null bounding boxes
    target_df = target_df.dropna(subset=['xmax', 'xmin', 'ymax', 'ymin'])
    nontarget_df = nontarget_df.dropna(subset=['xmax', 'xmin', 'ymax', 'ymin'])

    # if 'NC_1697554540' in str(csv_path):
    #     print(target_df)
    try:
        # Extract target bounding boxes as (xmin, ymin, xmax, ymax)
        target_boxes = target_df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    except Exception as e:
        print(csv_path)
        print(df)
        log.error(f"Error processing CSV: {csv_path}")
        log.error(e)
        exit(1)
    # Merge the target bounding boxes
    merged_boxes = merge_boxes(target_boxes, threshold=iou_threshold)

    # Create a DataFrame for the merged target boxes
    merged_target_df = pd.DataFrame(merged_boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])

    # Retain other relevant columns from the original target_df (like 'confidence' or 'class')
    other_columns = target_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1).reset_index(drop=True)
    merged_target_df = pd.concat([merged_target_df, other_columns], axis=1)

    # Combine the merged target rows with the original non-target rows
    final_df = pd.concat([merged_target_df, nontarget_df], ignore_index=True)

    # Save the updated CSV back to the same path
    final_df.to_csv(csv_path, index=False)
    log.info(f"Processed and saved merged CSV: {csv_path}")

def process_all_csvs_in_directory(directory_path, iou_threshold=0.5):
    """Process all CSV files in the given directory."""
    directory = Path(directory_path)

    # Find all CSV files in the directory
    csv_files = list(directory.rglob("*.csv"))
    log.info(f"Found {len(csv_files)} CSV files to process.")

    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file, iou_threshold)

def main(cfg: DictConfig) -> None:
    """Main function to process CSVs in a directory."""
    csv_directory = Path(cfg.batchdata.plant_dects, "processed")  # Directory containing CSV files
    iou_threshold = 0.5  # Default IoU threshold for merging

    # Process all CSVs in the specified directory
    process_all_csvs_in_directory(csv_directory, iou_threshold)
