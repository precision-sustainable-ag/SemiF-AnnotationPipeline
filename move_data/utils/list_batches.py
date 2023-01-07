import logging
import os
from pathlib import Path
from omegaconf import DictConfig
from utils.utils import read_keys
import pandas as pd

log = logging.getLogger(__name__)


class ListBatches:
    """Finds the specific data component and batch that has not been processed.
    Returns a list of batches and missing contents.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.keypath = cfg.pipeline_keys
        self.pkeys = read_keys(cfg.pipeline_keys)
        self.temp_path = Path(cfg.movedata.find_missing.container_list)

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
        self.temp_path.touch(exist_ok=True)
        # Developed image SAS key
        down_dev = self.pkeys.down_dev
        # Main command passed to os.system
        os.system(f"azcopy ls " + f'"{down_dev}"' +
                  f" | cut -d/ -f 1-{n} | awk '!a[$0]++' > {self.temp_path}")

    def organize_temp(self):
        """Reads, cleans, and writes back the 'azcopy list' results back to its 
        temporary txt file location."""
        # Read temporary file
        with open(self.temp_path, 'r') as f:
            lines = [line.rstrip() for line in f]
            # Strip azcopy list results strings
            lines = [x.replace("INFO: ", "") for x in lines]
            lines = [x.split(";")[0] for x in lines]
            lines = sorted(lines)
        # Write back to temp file location
        with open(self.temp_path, 'w') as f:
            for line in lines:
                f.write(f"{line}\n")

    def read_temp_results(self):
        """Reads temporary file created using 'azcopy list'"""
        with open(self.temp_path, 'r') as f:
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