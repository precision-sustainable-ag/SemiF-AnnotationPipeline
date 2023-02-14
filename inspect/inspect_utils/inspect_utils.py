import re
from symbol import eval_input
import pandas as pd
import os
import time
import textwrap
import cv2
import fnmatch
import signal
import logging
from pathlib import Path
from omegaconf import DictConfig

from inspect_utils.data_classes import InspectionSamples

log = logging.getLogger(__name__)


class CompileDataframe:

    def __init__(self, cfg: DictConfig) -> pd.DataFrame:
        self.cutout_dir = Path(cfg.data.cutoutdir)

        self.outdir = Path(cfg.inspect.outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)

        self.season_csv = cfg.inspect.season_csv
        self.processed_path = cfg.logs.processed
        self.overwrite_season_csv = cfg.inspect.overwrite_season_csv
        self.df = pd.DataFrame()

    def _read_processed_batches(self):
        with open(self.processed_path, 'r') as f:
            lines = [line.rstrip() for line in f]
            lines = (line for line in lines if line)  # Non-blank lines
        return sorted(lines)

    def _batches_contain_csvs(self):
        """
        1. Check existance of cutoutdir
        2. Gets cutout csvs based on batch ids in 'processed.txt' file
        3. Creates self.csvs (list of csv)
        4. Compares batches in processed.txt with availabel csvs, 
        """
        # Check cutout directory
        if self.cutout_dir.exists():
            # Read processed.txt
            batches = set(self._read_processed_batches())
        else:
            log.error(f"Cutout directory, {self.cutout_dir}, does not exist.")

        # Get csvs
        self.csvs = sorted([
            Path(self.cutout_dir, batch, batch + ".csv") for batch in batches
        ])

        [
            log.error(f"CSV {csv} does not exist.") for csv in self.csvs
            if not csv.exists()
        ]

        # Get csv stems to compare with batches in processed.txt
        csv_stems = set([Path(csv).stem for csv in self.csvs])
        # Compare batches with csvs
        if not batches == csv_stems:
            diff = csv_stems.symmetric_difference(batches)
            log.warning(
                f"Processed batches and available csvs do not match. Review {diff}."
            )

    def read_csvs(self):
        """Concatenates multiple csvs"""
        self._batches_contain_csvs()
        dfs = [pd.read_csv(x, low_memory=False) for x in self.csvs]
        df = pd.concat(dfs, ignore_index=True)
        return df

    def download_data(self):
        """TODO Download data from azure or from storage cluster if 
         cutout images don't exist"""
        pass

    def compile_df(self):
        df = self.read_csvs()
        self.df = df.copy()
        self.df = self.df[self.df["is_primary"] == True]
        self.df["state_id"] = self.df["batch_id"].str.split(
            "_", expand=True).iloc[:, 0]
        self.df["temp_cropout_path"] = self.df["batch_id"] + "/" + self.df[
            "cutout_path"].str.replace("png", "jpg")
        return self.df

    def save_cutout_csv(self):
        if Path(self.season_csv).is_file():
            if self.overwrite_season_csv:
                log.warning(
                    f"{self.season_csv} already exists. Overwriting copy.")
        self.df.to_csv(self.season_csv)


class CutoutStats:

    def __init__(self, cfg: DictConfig) -> pd.DataFrame:

        self.season_df = pd.read_csv(cfg.inspect.season_csv, low_memory=False)

    def cutout_counts(self):
        df = self.season_df.copy()
        # # exclude unknown and color checker
        df = df[(df["common_name"] != "colorchecker")
                & (df["common_name"] != "unknown")]

        # df_gb = df.groupby(["batch_id", "common_name"]).count()  #sample(n=20,
        df_gb = df.groupby(["batch_id", "common_name"]).sample(n=20,
                                                               random_state=42,
                                                               replace=True)
        # df_gb = stratified_sample(df, ["batch_id", "common_name"], size=.2)
        return df_gb

    def plot_stats(self):
        pass

    def compile_stats(self):
        pass

    def save_stats(self):
        pass

    def check_cutout_data(self):
        pass

    def strat_sample_cutouts(self):
        pass

    def create_sample_form(self):
        pass

    def compile_sample(self):
        pass

    def save_sample_data(self):
        pass


class ManualInspector:

    def __init__(self, cfg: DictConfig) -> None:
        self.csv = cfg.inspect.season_csv
        self.sample_size = cfg.inspect.visually_inspect.sample_size
        self.cutoutdir = str(Path(cfg.data.cutoutdir))
        self.finished_cuts = list()
        self.now = self.get_time()
        self.outdir = cfg.inspect.outdir
        self.df = pd.DataFrame()

        self.ex_dir = Path(self.outdir, "examples")
        self.ex_dir.mkdir(exist_ok=True, parents=True)

        self.save_path = None
        self.total_rows = 0
        self.idx = 0
        self.count = 0

        signal.signal(signal.SIGINT, self.sigint_handler)
        signal.signal(signal.SIGTERM, self.sigterm_handler)

    def sigint_handler(self, signum, frame):
        log.warning("\nPython SIGINT detected. Saving progress and exiting.\n")
        finished = pd.concat(self.finished_cuts)
        finished.to_csv(self.set_save_path())
        exit(1)

    def sigterm_handler(self, signum, frame):
        log.warning(
            "\nPython SIGTERM detected. Saving progress and exiting.\n")
        finished = pd.concat(self.finished_cuts)
        finished.to_csv(self.set_save_path(), index=False)
        exit(1)

    def find_match_csv(self):
        cont_csv = [
            x for x in Path(self.outdir).glob("*.csv")
            if f"__{self.idx} of {self.total_rows}__completed.csv" in x.name
        ]
        return cont_csv[0]

    def set_save_path(self):
        stem = f"inspection_results__{self.now}__{self.count} of {self.total_rows}__completed"
        if self.count == self.total_rows:
            stem = stem + "_FINAL"
        self.save_path = Path(self.outdir, stem + ".csv")
        return self.save_path

    def log_notes(self, batch_id):
        message = textwrap.fill(input(f"--{batch_id} notes: "), 50)
        log.info(message)

    def _read_cutout_csv(self):
        df = pd.read_csv(self.csv, low_memory=False)
        dfc = df.copy()
        df["state_id"] = dfc["batch_id"].str.split("_", expand=True).iloc[:, 0]
        return df

    def _get_sampling_data(self):
        self.df = self._read_cutout_csv()
        data = InspectionSamples(self.df, self.sample_size, self.cutoutdir)
        return data

    def get_time(self):
        os.environ['TZ'] = 'US/Central'
        time.tzset()
        t = time.localtime()
        now = time.strftime("%H:%M:%S %m-%d-%Y", t)
        return now

    def species_key(self, state_id):
        dfc = self.df[self.df["state_id"] == state_id]
        cnames = sorted(dfc["common_name"].unique())
        cnames.insert(0, 'weed')
        key_map = {v + 48: k for v, k in enumerate(cnames)}

        return key_map

    def inspect_gui(self):
        log.info("Starting inspection GUI...")

        data = self._get_sampling_data()

        mddf = data.md_sample
        ncdf = data.nc_sample
        mdlen = len(mddf)
        nclen = len(ncdf)
        self.total_rows = mdlen + nclen
        start = input("Start from the beginning? (y/n)")
        if start.lower() == "n":
            self.idx = int(
                input(f"Provide an index value from 0 to {mdlen + nclen}. "))

            # Get last inspection results
            prev_csv = self.find_match_csv()
            prev_df = pd.read_csv(prev_csv)
            # Add previous df to finished results
            self.finished_cuts.append(prev_df)
            # If MD has already been done
            if self.idx >= mdlen:
                nc_idx = self.idx - mdlen
                ncdfc = ncdf.copy()
                ncdf = ncdfc.iloc[nc_idx:]
                dfs = [ncdf]
            else:
                # If MD is not finished
                mddfc = mddf.copy()
                mddf = mddfc.iloc[self.idx:]
                dfs = [mddf, ncdf]
        else:
            dfs = [mddf, ncdf]
        self.count += self.idx
        batch_id = None
        old_batchid = None
        break_out_flag = False
        break_out_flag2 = False
        for df in dfs:
            kmap = self.species_key(df["state_id"].unique()[0])
            keys = {v: k for v, k in enumerate(kmap.values())}
            old_cname = None
            pred_cname = None
            for idx, row in df.iterrows():
                path = row["temp_cropout_path"]
                pred_cname = row["common_name"]
                if (old_cname is not None) and (old_cname != pred_cname):
                    print("-----------------------------------------")
                    print("------------ Species Change -------------")
                    print("-----------------------------------------")
                    time.sleep(2)
                    print("\nKeys: ")
                    for x, y in zip(keys.keys(), keys.values()):
                        print(f"{x}: {y}")
                old_cname = pred_cname
                batch_id = row["batch_id"]
                cutout_id = row["cutout_id"]

                print()
                if self.count % 500 == 0:
                    print(
                        f"Completed {self.count} of {self.total_rows} cutouts."
                    )
                print(batch_id)
                print(cutout_id)
                print("Prediction: ", pred_cname)

                if (old_batchid is not None) and (old_batchid != batch_id):
                    print("########################################")
                    print("############# Batch Change #############")
                    print("########################################")
                    time.sleep(3)
                    print(f"Just finished: {old_batchid}")
                    self.log_notes(old_batchid)
                old_batchid = batch_id

                img = cv2.imread(path)
                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image", 500, 500)
                cv2.imshow('image', img)
                cv2.setWindowTitle('image', pred_cname + " / " + cutout_id)
                result = None
                while result is None:
                    k = cv2.waitKey(0)

                    if k in range(47, 60):
                        result = list(kmap.values())[k - 48]

                    elif k == 42:
                        result = "colorchecker"

                    elif k == 45:
                        result = "unknown"

                    elif k == ord('n'):
                        self.log_notes(batch_id)

                    elif k == ord('h'):
                        keys = {v: k for v, k in enumerate(kmap.values())}
                        print("\nKeys: ")
                        for x, y in zip(keys.keys(), keys.values()):
                            print(f"{x}: {y}")
                        print("\nBatch: ", batch_id)
                        print("Cutout: ", cutout_id)
                        print("Prediction: ", pred_cname)

                    elif k == ord('s'):
                        print(f"Saving {self.count} inspection results.")
                        finished = pd.concat(self.finished_cuts)
                        finished.to_csv(self.set_save_path(), index=False)

                    elif k == ord('i'):
                        save_path = Path(self.ex_dir,
                                         cutout_id + f"_pred:{pred_cname}.png")
                        cv2.imwrite(str(save_path), img)
                        print(f"Saved {cutout_id}.")

                    elif k == 27:
                        break_out_flag = True
                        result = True
                        cv2.destroyAllWindows()
                        break

                    else:
                        print(k)

                print("Entered: ", result)
                row["true_species"] = result
                self.count += 1
                self.finished_cuts.append(row.to_frame().T)

                if break_out_flag:
                    break_out_flag2 = True
                    break
            if break_out_flag2:
                break
        log.info(f"Saving {self.count} inspection results.")
        finished = pd.concat(self.finished_cuts)
        finished.to_csv(self.set_save_path(), index=False)

    def graph_results(self):
        pass


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
