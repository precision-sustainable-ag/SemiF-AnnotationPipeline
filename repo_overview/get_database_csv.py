import datetime
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def concatenate_csv_from_folders_and_save(main_folder_path, output_file_path):
    batch_pref = ("MD", "TX", "NC")
    cutout_batches = [
        x for x in Path(main_folder_path).glob("*") if x.name.startswith(batch_pref)
    ]
    csv_files = [x for batch in cutout_batches for x in batch.glob("*.csv")]

    all_dfs = []

    for file_path in tqdm(csv_files, desc="Processing CSV files"):
        df = pd.read_csv(file_path, low_memory=False)

        # Ensure that the 'shape' column is of type string
        if "shape" in df.columns:
            df["shape"] = df["shape"].apply(lambda x: str(x) if not pd.isnull(x) else x)

        all_dfs.append(df)

    concatenated_df = pd.concat(all_dfs, ignore_index=True)

    # Save to Parquet format
    concatenated_df.to_parquet(output_file_path, index=False)

    return concatenated_df


def concat_parquets(longtermparquetpath, growdataparquetpath):
    df1 = pd.read_parquet(longtermparquetpath)
    print("Longterm Parquet loaded")
    df2 = pd.read_parquet(growdataparquetpath)
    print("GROWDATA Parquet loaded")
    df = pd.concat([df1, df2])
    print("DFs concatenated")
    datetimestr = Path(growdataparquetpath).stem.split("_")[-1]
    savedir = "database"
    name = f"concatenated_{datetimestr}.parquet"
    print(Path(savedir, name))
    df.to_parquet(Path(savedir, name), index=False)

    print("DF saved")
    print(Path(savedir, name).exists())





if __name__ == "__main__":
    # cutout_dir = "/mnt/research-projects/s/screberg/GROW_DATA/semifield-cutouts"
    # cutout_dir = "/mnt/research-projects/s/screberg/longterm_images/semifield-cutouts"
    longtermparquetpath = "database/updated_repo_LONGTERM_IMAGES_20240425.parquet"
    growdataparquetpath = "database/updated_repo_GROW_DATA_20240425.parquet"


    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    results_dir = Path("database/")
    results_dir.mkdir(exist_ok=True, parents=True)
    output_file_path = Path(results_dir, f"updated_repo_LONGTERM_IMAGES_{timestamp}.parquet")
    # large_df = concatenate_csv_from_folders_and_save(cutout_dir, str(output_file_path))
    concat_parquets(longtermparquetpath, growdataparquetpath)
