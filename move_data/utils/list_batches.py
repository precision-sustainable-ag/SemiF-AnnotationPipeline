import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.utils import read_keys

log = logging.getLogger(__name__)


class ListBatches:
    """Finds the specific data component and batch that has not been processed.
    Returns a list of batches and missing contents.
    """

    def __init__(self, cfg):
        self.pkeys = read_keys(cfg.pipeline_keys)
        self.container_list = Path(cfg.movedata.find_missing.container_list)
        log.info(f"Logging AZ blob container contents to {self.container_list}")

        self.pkeys = read_keys(cfg.movedata.SAS_keys)
        # Required data directories to be considered "processed".
        # Should be sub-directories of batch_id in Azure
        self.batch_data = cfg.movedata.find_missing.processed_data

    def az_list(self):
        """Passes 'azcopy list' command and writes to temporary txt file.
        Azcopy list command lists all the contents of the blob container
        and 2(n) levels of directories (ie the batch and batch data contents).

        """
        n = 2
        # Create temporary file if it doesn't exist
        self.container_list.touch(exist_ok=True)
        # Developed image SAS key
        down_dev = self.pkeys.down_dev
        # Main command passed to os.system
        os.system(
            f"azcopy ls "
            + f'"{down_dev}"'
            + f" | cut -d/ -f 1-{n} | awk '!a[$0]++' > {self.container_list}"
        )

    def organize_temp(self):
        """Reads, cleans, and writes back the 'azcopy list' results back to its
        temporary txt file location."""
        # Read temporary file
        with open(self.container_list, "r") as f:
            lines = [line.rstrip() for line in f]
            # Strip azcopy list results strings
            lines = [x.replace("INFO: ", "") for x in lines]
            lines = [x.split(";")[0] for x in lines]
            lines = sorted(lines)
        # Write back to temp file location
        with open(self.container_list, "w") as f:
            for line in lines:
                f.write(f"{line}\n")

    def read_temp_results(self):
        """Reads temporary file created using 'azcopy list'"""

        with open(self.container_list, "r") as f:
            lines = [line.rstrip() for line in f]
        return lines

    def temp2df(self):
        """Creates dataframe from temporary txt file.
        Splits 'azcopy list' results by '/' to create 'batch' and 'child'
        columns.

        Returns:
            df(dataframe): dataframe with 'batch' and 'child' columns
        """
        lines = self.read_temp_results()
        res = pd.DataFrame(lines, columns=["result"])
        exp_df = res.result.str.split("/", expand=True).apply(pd.Series)
        exp_df.columns = ["batch", "child"]
        concat_df = pd.concat([res, exp_df], axis=1)
        df = concat_df[["batch", "child"]]
        df = df.dropna()
        return df

    def find_missing(self):
        """Compares batch data directories in azure with a list of required batch directories.
        Returns a dataframe of batches and their missing data products in Azure blob container.

        Returns:
            df(dataframe): dataframe of 'batch' and 'missing' columns
        """
        # Get df from temp list
        temp = self.temp2df()

        dfs = []
        # Loop through unique batch dataframes
        for ubatch in temp.batch.unique():
            udf = temp[temp["batch"] == ubatch]
            data_list = list(udf.child)
            # Find missing data
            missing = []
            for i in self.batch_data:
                # Main comparison line
                res = any(i in sub for sub in data_list)
                if not res:
                    missing.append(i)
            # Create missing df
            if len(missing) > 0:
                ndf = pd.DataFrame()
                ndf["missing"] = sorted([missing])
                ndf["batch"] = ubatch
                ndf = ndf[["batch", "missing"]]
                dfs.append(ndf)

        df = pd.concat(dfs).reset_index(drop=True)
        return df


class BatchProcessor:
    def __init__(self, cfg):

        """Summarizes the processed and unprocessed batches by filtering the contents of container_contents.txt

        Args:
            container_list (str): file path to container_list.txt
        """
        self.container_list = cfg.movedata.find_missing.container_list
        self.folders = self.read_container_list(self.container_list)
        # Sub-folders that must be present to be considered "processed"
        self.necessary_subfolders = ["autosfm", "meta_masks", "metadata"]
        self.results = {}
        # Extract unique batches and states
        self.batches = set(folder.split("/")[0] for folder in self.folders if folder)
        self.states = set(batch.split("_")[0] for batch in self.batches if "_" in batch)
        # Date ranges for each "season" by state
        self.date_ranges = {
            "MD": {
                "weeds 2022": ("2022-03-04", "2022-09-17"),
                "cover crops 2022/2023": ("2022-10-20", "2023-04-25"),
                "cash crops 2023": ("2023-05-12", "2023-07-07"),
                "weeds 2023": ("2023-07-17", "2023-09-17"),
                "cover crops 2023/2024": ("2023-09-18", "2024-05-16"),
                "weeds 2024": ("2024-07-08", "2024-09-08"),
            },
            "NC": {
                "weeds 2022": ("2022-03-04", "2022-09-23"),
                "cover crops 2022/2023": ("2022-10-11", "2023-03-30"),
                "cash crops 2023": ("2023-06-06", "2023-07-25"),
                "weeds 2023": ("2023-08-21", "2023-10-10"),
                "cover crops 2023/2024": ("2023-10-11", "2024-03-14"),
                "weeds 2024": ("2024-07-08", "2024-10-08"),
            },
            "TX": {
                "weeds 2022": ("2022-03-04", "2022-10-24"),
                "cover crops 2022/2023": ("2022-10-25", "2023-04-04"),
                "cash crops 2023": ("2023-05-08", "2023-08-01"),
                "weeds 2023": ("2023-08-04", "2023-10-19"),
                "cover crops 2023/2024": ("2023-10-20", "2024-03-01"),
                "weeds 2024": ("2024-03-26", "2024-10-26"),
            },
        }

    # existing __init__ and other methods remain unchanged

    def save_results_to_csv(self, csv_filename):
        """
        Saves the summary of processed and not processed batches to a CSV file.

        Args:
            csv_filename (str): The filename for the output CSV.
        """
        # Initialize a list to store the rows of the DataFrame
        rows = []

        # Iterate over states, categories, and their respective statuses
        for state, categories in self.results.items():
            for category, status_data in categories.items():
                for status in ["Processed", "Not Processed"]:
                    for batch in status_data[status]:
                        # Append a row for each batch
                        rows.append(
                            {
                                "state": state,
                                "category": category,
                                "batch": batch,
                                "status": status,
                            }
                        )

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows, columns=["state", "category", "batch", "status"])

        # Save the DataFrame to a CSV file
        df.to_csv(csv_filename, index=False)

        print(f"Results saved to {csv_filename}")

    def read_container_list(self, path):
        with open(path) as f:
            folders = [line.rstrip("\n") for line in f]
        return folders

    def determine_status(self):
        for state, categories in self.date_ranges.items():
            self.results[state] = {}
            # Build a dictionary of dates for each state
            dates_by_state = {
                state: sorted(
                    [
                        batch.split("_")[1]
                        for batch in self.batches
                        if batch.startswith(state)
                    ]
                )
                for state in self.states
            }
            for category, (start_date, end_date) in categories.items():
                self.results[state][category] = {
                    "Processed": [],
                    "Not Processed": [],
                }
                # for batch_date in dates_by_state:
                for batch_date in dates_by_state[state]:
                    batch_name = f"{state}_{batch_date}"
                    if start_date <= batch_date <= end_date:
                        is_processed = all(
                            f"{batch_name}/{subfolder}" in self.folders
                            for subfolder in self.necessary_subfolders
                        )
                        status = "Processed" if is_processed else "Not Processed"
                        self.results[state][category][status].append(batch_name)

    def print_results(self):
        for state, categories in self.results.items():
            print(f"\n-----------------------------")
            print(f"State: {state}")
            print(f"-----------------------------")
            for category, status_data in categories.items():
                print(f"\n{category.title()}")

                # Processed
                print("\nProcessed:")
                for batch in status_data["Processed"]:
                    print(f"  - {batch}")

                # Not Processed
                print("\nNot Processed:")
                for batch in status_data["Not Processed"]:
                    print(f"  - {batch}")

    def save_to_file(self, filename):
        with open(filename, "w") as file:
            for state, categories in self.results.items():
                file.write(f"\n-----------------------------\n")
                file.write(f"State: {state}\n")
                file.write(f"-----------------------------\n")
                for category, status_data in categories.items():
                    file.write(f"\n{category.title()}\n")

                    # Processed
                    file.write("\nProcessed:\n")
                    for batch in status_data["Processed"]:
                        file.write(f"  - {batch}\n")

                    # Not Processed
                    file.write("\nNot Processed:\n")
                    for batch in status_data["Not Processed"]:
                        file.write(f"  - {batch}\n")

    def write_summary(self, filename):
        with open(filename, "a") as file:
            file.write("\n\n===============================\n")
            file.write("Summary:\n")
            file.write("===============================\n")
            for state, categories in self.results.items():
                file.write(f"State: {state}\n")
                for category, status_data in categories.items():
                    processed_count = len(status_data["Processed"])
                    not_processed_count = len(status_data["Not Processed"])
                    file.write(
                        f"Category: {category} - Processed: {processed_count}, Not Processed: {not_processed_count}\n"
                    )
                file.write("-----------------------------\n")
