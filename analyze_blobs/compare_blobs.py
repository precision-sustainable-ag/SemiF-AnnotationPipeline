import csv
import logging
import re
import sys

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)
from datetime import datetime

from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

from utils.utils import read_keys


class AzureBlobFolderComparer:
    def __init__(self, cfg):
        self.keys = read_keys(cfg.pipeline_keys)
        self.account_url = self.keys.account_url
        self.container_name_1 = cfg.blob_names.upload
        self.container_name_2 = cfg.blob_names.developed
        # self.container_name_2 = cfg.blob_names.cutout

        self.date_ranges = cfg.date_ranges
        self.folder_pattern = re.compile(r"^[A-Z]{2}_\d{4}-\d{2}-\d{2}(?![\w-])")
        self.sas_token_1 = self.keys.down_upload.split("?")[1]
        self.sas_token_2 = self.keys.down_dev.split("?")[1]
        # self.sas_token_2 = self.keys.down_cut.split("?")[1]

    def list_unique_folders(self, container_name, sas_token):
        """List all unique folders in a given container using walk_blobs."""
        blob_service_client = BlobServiceClient(
            account_url=self.account_url, credential=sas_token
        )
        container_client = blob_service_client.get_container_client(container_name)
        unique_folders = set()

        for blob in tqdm(container_client.walk_blobs()):
            folder_name = blob.name.strip("/")
            if folder_name and self.folder_pattern.match(folder_name):
                if " " in folder_name:
                    folder_name = folder_name.split(" ")[0]
                unique_folders.add(folder_name)
        return unique_folders

    def classify_folders(self, folders):
        """Classify folders based on state and date ranges."""
        classified_folders = {
            state: {period: [] for period in self.date_ranges[state]}
            for state in self.date_ranges
        }

        for folder in folders:
            state, date_str = folder.split("_")

            folder_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            for period in self.date_ranges[state].keys():
                start = self.date_ranges[state][period]["start"]
                end = self.date_ranges[state][period]["end"]

                start_date = datetime.strptime(start, "%Y-%m-%d").date()
                end_date = datetime.strptime(end, "%Y-%m-%d").date()

                if start_date <= folder_date <= end_date:
                    classified_folders[state][period].append(folder)
                    break

        return classified_folders

    def find_missing_folders(self, classified_folders_src, classified_folders_tgt):
        """Find missing folders within date ranges."""
        missing_folders = {
            state: {period: [] for period in self.date_ranges[state]}
            for state in self.date_ranges
        }

        for state, periods in classified_folders_src.items():
            for period, folders in periods.items():
                missing_folders[state][period] = sorted(
                    set(folders)
                    - set(classified_folders_tgt.get(state, {}).get(period, []))
                )
        return missing_folders

    @staticmethod
    def count_items_in_nested_dict(nested_dict):
        count_dict = {}
        for state, ranges in nested_dict.items():
            count_dict[state] = {}
            for range_name, items in ranges.items():
                count_dict[state][range_name] = len(items)
        return count_dict

    def total_batch(self, missing_folders):
        """Calculate the percentage of completed and missing batches."""
        total_batches = len(
            self.list_unique_folders(self.container_name_1, self.sas_token_1)
        )

        missing_batches = len(missing_folders)

        if total_batches == 0:
            return 0, 0

        completed = total_batches - missing_batches
        completed_percentage = (completed / total_batches) * 100
        missing_percentage = (missing_batches / total_batches) * 100

        return (
            total_batches,
            completed,
            missing_batches,
            completed_percentage,
            missing_percentage,
        )

    def write_missing_folders_to_file(self, missing_folders, file_path):
        missing_folders_counts = self.count_items_in_nested_dict(missing_folders)
        # Calculating the total missing folders and total folders
        total_missing_folders = sum(
            folders
            for state_data in missing_folders_counts.values()
            for folders in state_data.values()
        )
        total_batches = len(
            self.list_unique_folders(self.container_name_1, self.sas_token_1)
        )
        """Write missing folders to a text file."""
        with open(file_path, "w") as file:
            # Write the report at the top of the file
            header_msg = "A list of batches that exist in the semif-upload \nbut not in the semif-developed-images blob container."
            file.write(f"{header_msg}\n\n")
            file.write(
                f"-----------------Missing Batches Summary Report-------------------\n"
            )
            file.write(f"Total batches in semif-upload: {total_batches}\n")
            file.write(
                f"Total missing batches not in semif-developed-images: {total_missing_folders}\n\n"
            )
            for state, periods in missing_folders_counts.items():
                file.write(f"State: {state}\n")
                for period, folders_count in periods.items():
                    file.write(f"  {period}: {folders_count}\n")
            file.write("\n")
            file.write(
                f"----------------Missing Batches by State and Season--------------\n"
            )
            for state, periods in missing_folders.items():
                file.write(f"State: {state}\n")
                for period, folders in periods.items():
                    file.write(f"  {period}:\n")
                    for folder in folders:
                        file.write(f"    - {folder}\n")
                file.write("\n")

    def write_missing_folders_to_csv(self, missing_folders, csv_file_path):
        """Write missing folders to a CSV file."""
        with open(csv_file_path, mode="w", newline="") as file:
            csv_writer = csv.writer(file)
            # Write CSV headers
            csv_writer.writerow(["index", "state", "season", "batch_id"])

            # Initialize index
            index = 1
            # Write data rows
            for state, periods in missing_folders.items():
                for season, batch_ids in periods.items():
                    for batch_id in batch_ids:
                        csv_writer.writerow([index, state, season, batch_id])
                        index += 1


# Usage
@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # fmt: off
    comparer = AzureBlobFolderComparer(cfg)
    folders_src = comparer.list_unique_folders(container_name=comparer.container_name_1, sas_token=comparer.sas_token_1)
    folders_tgt = comparer.list_unique_folders(container_name=comparer.container_name_2, sas_token=comparer.sas_token_2)
    
    classified_folders_src = comparer.classify_folders(folders_src)
    classified_folders_tgt = comparer.classify_folders(folders_tgt)

    missing_folders = comparer.find_missing_folders(classified_folders_src, classified_folders_tgt)
    comparer.write_missing_folders_to_file(missing_folders, cfg.logs.preprocessed_backlog)
    comparer.write_missing_folders_to_csv(missing_folders, cfg.logs.preprocessed_backlog.replace(".txt", ".csv"))
