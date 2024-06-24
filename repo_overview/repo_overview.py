import datetime
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import seaborn as sns
from omegaconf import DictConfig


class SampleImageData:
    def __init__(self, cfg, df):
        self.df = df
        self.longterm_storage = cfg.data.longterm_storage
        self.cutout_n = 5  # per species
        self.repo_sample_images = Path(cfg.data.repo_database, "results", "images")
        self.repo_sample_images.mkdir(parents=True, exist_ok=True)

    def sample_images_by_species(self):
        pass

    def sample_cutouts_by_species(self):
        samp_dfs = []
        for common_name in self.df.common_name.unique():
            if "morning" in common_name.lower():
                print(common_name)
            if "morning" not in common_name.lower():
                continue
            cndf = self.df[self.df["common_name"] == common_name]
            cuts = cndf[cndf["green_sum"] > 10000]
            if len(cuts) == 0:
                cuts = cndf[cndf["green_sum"] > 1000]
                if len(cuts) == 0:
                    cuts = cndf[cndf["green_sum"] > 100]

            samp_cuts = cuts.sample(n=5, replace=True)
            samp_cuts["cropout_path"] = (
                self.longterm_storage
                + "/semifield-cutouts/"
                + samp_cuts["batch_id"]
                + "/"
                + samp_cuts["cutout_id"]
                + ".jpg"
            )
            samp_cuts["cutout_path"] = (
                self.longterm_storage
                + "/semifield-cutouts/"
                + samp_cuts["batch_id"]
                + "/"
                + samp_cuts["cutout_id"]
                + ".png"
            )
            samp_dfs.append(samp_cuts)

        df = pd.concat(samp_dfs)
        return df

    def copy_cutouts(self):
        df = self.sample_cutouts_by_species()
        for _, row in df.iterrows():
            cutout_src = row["cutout_path"]
            cropout_src = row["cropout_path"]
            common_name = row["common_name"]
            dst_dir = Path(self.repo_sample_images, common_name)
            dst_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy2(cutout_src, dst_dir)
            shutil.copy2(cropout_src, dst_dir)


class ParquetDataProcessor:
    def __init__(self, cfg):
        self.database_parquet = cfg.data.database_parquet
        self.repo_database = cfg.data.repo_database
        # self.db_path = self.get_most_recent_parquet_file()
        self.db_path = Path(
            "/home/psa_images/SemiF-AnnotationPipeline/repo_overview/database/concatenated_20240425.parquet"
        )
        self.setup_data_results_dirs()
        self.df = None
        self.get_formatted_season2datetime(dict(cfg.date_ranges))

    def get_most_recent_parquet_file(self):
        """
        Finds the most recent file in a directory where files have a timestamp in their name.

        :param directory: Directory to search in.
        :return: The path to the most recent file.
        """
        most_recent_file = None
        latest_timestamp = None

        for file in os.listdir(self.database_parquet):
            if file.endswith("parquet"):
                parts = file.split("_")
                if len(parts) > 1:
                    timestamp_str = parts[-1].split(".")[0]  # Extract timestamp part
                    try:
                        # Convert timestamp string to datetime object
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d")
                        if not latest_timestamp or timestamp > latest_timestamp:
                            latest_timestamp = timestamp
                            most_recent_file = file
                    except ValueError:
                        # If the timestamp is not in the expected format, ignore the file
                        continue

        return (
            Path(self.database_parquet, most_recent_file)
            if most_recent_file
            else "No recent file found."
        )

    def setup_data_results_dirs(self):
        timestamp = datetime.now().strftime("%Y%m%d")
        results_dir = Path(self.repo_database, "results", f"data_{timestamp}")

        self.metrics_results_dir = Path(results_dir, "metrics")
        self.metrics_results_dir.mkdir(parents=True, exist_ok=True)

        self.figs_results_dir = Path(results_dir, "figs")
        self.figs_results_dir.mkdir(parents=True, exist_ok=True)

    def get_formatted_season2datetime(self, seasons_dict):
        self.seasons_dict = {
            state: {
                season: {
                    "start": datetime.strptime(dates["start"], "%Y-%m-%d"),
                    "end": datetime.strptime(dates["end"], "%Y-%m-%d"),
                }
                for season, dates in seasons.items()
            }
            for state, seasons in seasons_dict.items()
        }

    @staticmethod
    def capitalize_first_letter(string):
        return string.capitalize() if string else string

    @staticmethod
    def add_season_column(df, date_col, state_col, seasons_dict):
        # Function to determine the season for a given date and state
        def get_season(date, state):
            for season, dates in seasons_dict[state].items():
                if dates["start"] <= date <= dates["end"]:
                    return season
            return None

        # Convert the date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        # Apply the get_season function to each row
        df["season"] = df.apply(
            lambda row: get_season(row[date_col], row[state_col]), axis=1
        )

        return df

    def read_parquet_file(self, totals=True):
        self.df = pd.read_parquet(self.db_path)
        self.df["state_id"] = self.df.batch_id.str.split("_", expand=False).str[0]
        self.df["date"] = self.df.batch_id.str.split("_", expand=False).str[1]
        self.df["dt"] = pd.to_datetime(self.df["date"])
        self.df["common_name"] = self.df["common_name"].apply(
            self.capitalize_first_letter
        )
        self.df = self.add_season_column(self.df, "dt", "state_id", self.seasons_dict)
        self.df["general_season"] = self.df["season"].apply(self.assign_season)

        self.df = self.fix_cash_crops(self.df)
        self.df = self.fix_weeds(self.df)
        if totals:
            self.total_species = self.df[self.df["common_name"] != "Colorchecker"][
                "common_name"
            ].nunique()
            self.image_total = self.df["image_id"].sum()

    def fix_weeds(self, df):
        df.loc[df.common_name == "Horseweed", "general_season"] = "weeds"
        return df

    def fix_cash_crops(self, df):
        df.loc[df.common_name == "Soybean", "general_season"] = "cash crops"
        df.loc[df.common_name == "Upland cotton", "general_season"] = "cash crops"
        df.loc[df.common_name == "Maize", "general_season"] = "cash crops"
        return df

    def image_by_common_name(self):
        # sdf = self.df.drop_duplicates("image_id")
        sdf = self.df.copy()
        sdf = sdf[sdf["common_name"] != "Colorchecker"]
        cdf = (
            sdf.groupby(["common_name"])
            .image_id.nunique()
            .reset_index()
            .sort_values("image_id")
        )

        cdf.to_csv(f"{self.metrics_results_dir}/images_by_common_name.csv")

    def table_for_pub(self):
        # sdf = self.df.drop_duplicates("image_id")
        sdf = self.df.copy()
        sdf = sdf[sdf["common_name"] != "Colorchecker"]
        cdf = (
            sdf.groupby(["common_name", "general_season"])
            .image_id.nunique()
            .reset_index()
            .sort_values(["general_season", "common_name"])
        )

        cdf.to_csv(f"{self.metrics_results_dir}/table_for_pub.csv", index=False)

    def images_by_common_name_state(self, save_fig=False):
        sdf = self.df.copy()  # self.df.drop_duplicates("image_id")
        sdf = sdf[sdf["common_name"] != "Colorchecker"]
        csdf = (
            sdf.groupby(["common_name", "state_id"])
            .image_id.nunique()
            .reset_index()
            .sort_values("image_id")
        )
        csdf.to_csv(f"{self.metrics_results_dir}/images_by_common_name_and_state.csv")
        if save_fig:
            # print(sdf)
            sns.set_context("notebook")
            sns.set(font_scale=2)
            g = sns.catplot(
                data=csdf,
                x="image_id",
                y="common_name",
                orient="horizontal",
                col="state_id",
                sharey=False,
                kind="bar",
                height=10,
            )
            g.set(xlabel="", ylabel="")
            g.set_titles(col_template="{col_name}")
            g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
            g.fig.suptitle("Full Bench Images by Species")
            for ax in g.axes.ravel():
                # add annotations
                for c in ax.containers:
                    # labels = [f'{(v.get_width() / 1000):.1f}K' for v in c]
                    labels = [int(v.get_width()) for v in c]
                    ax.bar_label(c, labels=labels, label_type="edge", fontsize=14)
                ax.margins(y=0.2)
            g.savefig(
                f"{self.figs_results_dir}/images_by_common_name_and_state.png",
                dpi=150,
            )

    def cutouts_by_common_name(self):
        cdf = self.df.drop_duplicates("cutout_id")
        cdf = cdf[cdf["common_name"] != "Colorchecker"].reset_index(drop=True)
        cdf = (
            cdf.groupby(["common_name"])
            .cutout_id.count()
            .reset_index()
            .sort_values("cutout_id")
        )

        cdf.to_csv(f"{self.metrics_results_dir}/cutouts_by_common_name.csv")
        self.cutout_total = cdf["cutout_id"].sum()

    def cutouts_by_common_name_state_and_season(self, save_fig=False):
        cdf = self.df.drop_duplicates("cutout_id")
        cdf = (
            self.df.groupby(["common_name", "state_id", "general_season"])
            .cutout_id.count()
            .reset_index()
            .sort_values("cutout_id")
        )

        cdf = cdf[cdf["common_name"] != "Colorchecker"].reset_index(drop=True)
        print(cdf)



    def cutouts_by_common_name_state(self, save_fig=False):
        cdf = self.df.drop_duplicates("cutout_id")
        cdf = (
            self.df.groupby(["common_name", "state_id"])
            .cutout_id.count()
            .reset_index()
            .sort_values("cutout_id")
        )
        cdf = cdf[cdf["common_name"] != "Colorchecker"].reset_index(drop=True)
        cdf.to_csv(f"{self.metrics_results_dir}/cutouts_by_common_name_and_state.csv")
        if save_fig:
            # print(sdf)
            sns.set_context("notebook")
            sns.set_theme(font_scale=2)
            g = sns.catplot(
                data=cdf,
                x="cutout_id",
                y="common_name",
                orient="horizontal",
                col="state_id",
                sharey=False,
                kind="bar",
                height=10,
            )
            g.set(xlabel="", ylabel="")
            g.set_titles(col_template="{col_name}")
            g.figure.subplots_adjust(top=0.9)  # adjust the Figure in rp
            g.figure.suptitle("Full Bench Cutouts by Species and State")
            for ax in g.axes.ravel():
                # add annotations
                for c in ax.containers:
                    # labels = [f'{(v.get_width() / 1000):.1f}K' for v in c]
                    labels = [int(v.get_width()) for v in c]
                    ax.bar_label(c, labels=labels, label_type="edge", fontsize=14)
                ax.margins(y=0.2)
            g.savefig(
                f"{self.figs_results_dir}/cutouts_by_common_name_and_state.png",
                dpi=150,
            )

    def primary_cutouts_by_common_name_general_season(self, save_fig=False):
        pcdf = self.df[self.df["is_primary"] == True]
        gb_pcdf = (
            pcdf.groupby(["general_season", "common_name"])
            .cutout_id.count()
            .reset_index()
            .sort_values(["general_season", "cutout_id"])
        )
        fin_gb_pcdf = gb_pcdf[gb_pcdf["common_name"] != "Colorchecker"].reset_index(
            drop=True
        )
        fin_gb_pcdf.to_csv(
            f"{self.metrics_results_dir}/primary_cutouts_by_common_name_general_season.csv"
        )

        if save_fig:
            sns.set_context("notebook")
            sns.set_theme(font_scale=2)
            g = sns.catplot(
                data=fin_gb_pcdf,
                x="cutout_id",
                y="common_name",
                orient="horizontal",
                col="general_season",
                sharey=False,
                kind="bar",
                height=14,
            )
            g.set(xlabel="", ylabel="")
            g.set_titles(col_template="{col_name}")
            g.figure.subplots_adjust(top=0.9)  # adjust the Figure in rp
            g.figure.suptitle("Cutouts by Species, State, and Primary Status")
            for ax in g.axes.ravel():
                # add annotations
                for c in ax.containers:
                    # labels = [f'{(v.get_width() / 1000):.1f}K' for v in c]
                    labels = [int(v.get_width()) for v in c]
                    ax.bar_label(c, labels=labels, label_type="edge", fontsize=14)
                ax.margins(y=0.2)
            g.savefig(
                f"{self.figs_results_dir}/primary_cutouts_by_common_name_general_season.png",
                dpi=150,
            )

    def primary_weed_cutouts_by_common_nameEPPO_state(self, save_fig=False):
        pcdf = self.df[self.df["is_primary"] == True]
        pcdf = self.df[self.df["general_season"] == "weeds"]
        pcdf = pcdf[pcdf["common_name"] != "Unknown"]
        pcdf = pcdf[pcdf["common_name"] != "Colorchecker"].reset_index(drop=True)

        pcdf["common_name/EPPO"] = pcdf["common_name"] + "/" + pcdf["EPPO"]

        fin_gb_pcdf = (
            pcdf.groupby(["state_id", "common_name/EPPO"])
            .cutout_id.count()
            .reset_index()
            .sort_values(["state_id", "cutout_id"])
        )

        fin_gb_pcdf.to_csv(
            f"{self.metrics_results_dir}/primary_weed_cutouts_by_common_nameEPPO_state.csv"
        )

        if save_fig:
            sns.set_context("notebook")
            sns.set_theme(font_scale=2)
            g = sns.catplot(
                data=fin_gb_pcdf,
                x="cutout_id",
                y="common_name/EPPO",
                orient="horizontal",
                col="state_id",
                sharey=False,
                kind="bar",
                height=14,
            )
            g.set(xlabel="", ylabel="")
            g.set_titles(col_template="{col_name}")
            g.figure.subplots_adjust(top=0.9)  # adjust the Figure in rp
            g.figure.suptitle("Primary Cutouts by Common Name/EPPO and state")
            for ax in g.axes.ravel():
                # add annotations
                for c in ax.containers:
                    # labels = [f'{(v.get_width() / 1000):.1f}K' for v in c]
                    labels = [int(v.get_width()) for v in c]
                    ax.bar_label(c, labels=labels, label_type="edge", fontsize=14)
                ax.margins(y=0.2)
            g.savefig(
                f"{self.figs_results_dir}/primary_weed_cutouts_by_common_nameEPPO_state.png",
                dpi=150,
            )

    def primary_cutouts_by_common_name(self):
        pcdf = self.df[self.df["is_primary"] == True]
        gb_pcdf = (
            pcdf.groupby(["common_name"])
            .cutout_id.count()
            .reset_index()
            .sort_values("cutout_id")
        )
        fin_gb_pcdf = gb_pcdf[gb_pcdf["common_name"] != "Colorchecker"].reset_index(
            drop=True
        )
        fin_gb_pcdf.to_csv(
            f"{self.metrics_results_dir}/primary_cutouts_by_common_name.csv"
        )
        self.total_primary_cutouts = fin_gb_pcdf["cutout_id"].sum()

    def primary_status_cutouts_by_common_name_state(self, save_fig=False):
        gb_pcdf = (
            self.df.groupby(["common_name", "state_id", "is_primary"])
            .cutout_id.count()
            .reset_index()
            .sort_values("cutout_id")
        )
        fin_gb_pcdf = gb_pcdf[gb_pcdf["common_name"] != "Colorchecker"].reset_index(
            drop=True
        )
        fin_gb_pcdf.to_csv(
            f"{self.metrics_results_dir}/primary_status_cutouts_by_common_name_and_state.csv"
        )
        if save_fig:
            sns.set_context("notebook")
            sns.set_theme(
                context="talk",  # one of {paper, notebook, talk, poster}
                style="whitegrid",  # one of {darkgrid, whitegrid, dark, white, ticks}
                font="sans-serif",
                font_scale=2,
            )
            # sns.set_theme(font_scale=2)
            g = sns.catplot(
                data=fin_gb_pcdf[fin_gb_pcdf["is_primary"] == True],
                x="cutout_id",
                y="common_name",
                orient="horizontal",
                col="state_id",
                sharey=False,
                kind="bar",
                height=14,
            )
            g.set(xlabel="", ylabel="")
            g.set_titles(col_template="{col_name}")
            g.figure.subplots_adjust(top=0.9)  # adjust the Figure in rp
            g.figure.suptitle("Cutouts by Species, State, and Primary Status")
            for ax in g.axes.ravel():
                # add annotations
                for c in ax.containers:
                    # labels = [f'{(v.get_width() / 1000):.1f}K' for v in c]
                    labels = [int(v.get_width()) for v in c]
                    ax.bar_label(c, labels=labels, label_type="edge", fontsize=14)
                ax.margins(y=0.2)
            g.savefig(
                f"{self.figs_results_dir}/cutouts_by_common_name_state_isprimary_TRUE.png",
                dpi=150,
            )

    # Function to determine the food type
    @staticmethod
    def assign_season(season):
        if season in ["weeds 2023", "weeds 2022"]:
            return "weeds"
        elif season in ["cover crops 2022/2023", "cover crops 2023/2024"]:
            return "cover crops"
        elif season in ["cash crops 2023", "cash crops 2024"]:
            return "cash crops"
        else:
            return "other"

    def cutouts_by_species_and_season(self, save_fig=False):
        cs_df = self.df.copy()

        fin_cs_df = cs_df[cs_df["common_name"] != "Colorchecker"].reset_index(drop=True)
        # Applying the function to create the new column
        fin_cs_df["general_season"] = fin_cs_df["season"].apply(self.assign_season)
        fin_cs_df = fin_cs_df.drop(columns=["season"])

        fin_cs_df.loc[fin_cs_df.common_name == "Maize", "general_season"] = "cash crops"
        fin_cs_df.loc[fin_cs_df.common_name == "Soybean", "general_season"] = (
            "cash crops"
        )

        fin_cs_df = (
            fin_cs_df.groupby(["general_season", "common_name"])
            .cutout_id.count()
            .reset_index()
            .sort_values("cutout_id")
        )

        fin_cs_df = fin_cs_df[["general_season", "common_name", "cutout_id"]]
        fin_cs_df.to_csv(
            f"{self.metrics_results_dir}/cutouts_by_species_and_season.csv"
        )
        if save_fig:
            # print(sdf)
            sns.set_context("notebook")
            sns.set(font_scale=1.5)
            g = sns.catplot(
                data=fin_cs_df,
                x="cutout_id",
                y="common_name",
                orient="horizontal",
                col="general_season",
                sharey=False,
                kind="bar",
                errorbar=None,
                height=15,
            )
            g.set(xlabel="", ylabel="")
            g.set_titles(col_template="{col_name}")
            g.fig.subplots_adjust(
                top=0.9
            )  # adjust the bottom to give more space for labels
            g.fig.suptitle("Cutouts by species and season")
            for ax in g.axes.ravel():
                # add annotations
                for c in ax.containers:
                    # labels = [f'{(v.get_width() / 1000):.1f}K' for v in c]
                    labels = [int(v.get_width()) for v in c]
                    ax.bar_label(c, labels=labels, label_type="edge", fontsize=22)
                ax.margins(y=0.2)
            for ax in g.axes.ravel():
                ax.get_xaxis().set_visible(False)
            g.savefig(
                f"{self.figs_results_dir}/cutouts_by_species_and_season.png",
                dpi=150,
                bbox_inches="tight",
            )

    def images_by_species_and_season(self, save_fig=False):
        igb_df = self.df.copy()
        igb_df["general_season"] = igb_df["season"].apply(self.assign_season)
        igb_df.loc[igb_df.common_name == "Maize", "general_season"] = "cash crops"
        igb_df.loc[igb_df.common_name == "Soybean", "general_season"] = "cash crops"
        igb_df = igb_df.drop(columns=["season"])
        igb_df = igb_df[igb_df["common_name"] != "Colorchecker"].reset_index(drop=True)
        fin_igb_df = (
            igb_df.groupby(["general_season", "common_name"])
            .image_id.nunique()
            .reset_index()
            .sort_values("image_id")
        )

        fin_igb_df.to_csv(
            f"{self.metrics_results_dir}/images_by_species_and_season.csv"
        )
        if save_fig:
            # print(sdf)
            sns.set_context("notebook")
            sns.set(font_scale=1.5)
            g = sns.catplot(
                data=fin_igb_df,
                x="image_id",
                y="common_name",
                orient="horizontal",
                col="general_season",
                sharey=False,
                kind="bar",
                errorbar=None,
                height=15,
            )
            g.set(xlabel="", ylabel="")
            g.set_titles(col_template="{col_name}")
            g.fig.subplots_adjust(
                top=0.9
            )  # adjust the bottom to give more space for labels
            g.fig.suptitle("Images by species and season")
            for ax in g.axes.ravel():
                # add annotations
                for c in ax.containers:
                    # labels = [f'{(v.get_width() / 1000):.1f}K' for v in c]
                    labels = [int(v.get_width()) for v in c]
                    ax.bar_label(c, labels=labels, label_type="edge", fontsize=14)
                ax.margins(y=0.2)
            for ax in g.axes.ravel():
                ax.get_xaxis().set_visible(False)
            g.savefig(
                f"{self.figs_results_dir}/images_by_species_and_season.png",
                dpi=150,
                bbox_inches="tight",
            )

    def unique_batches_by_state_season(self, save_fig=False):
        bdf = self.df.drop_duplicates("batch_id")
        bdf = bdf.groupby(["state_id", "season"])["batch_id"].count().reset_index()
        bdf.to_csv(f"{self.metrics_results_dir}/batches_by_state_and_season.csv")

        self.batch_total = bdf["batch_id"].sum()
        if save_fig:
            # print(sdf)
            sns.set_context("notebook")
            sns.set(font_scale=2)
            g = sns.catplot(
                data=bdf,
                x="batch_id",
                y="season",
                orient="horizontal",
                col="state_id",
                sharey=False,
                kind="bar",
                height=10,
            )
            g.set(xlabel="", ylabel="")
            g.set_titles(col_template="{col_name}")
            g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
            g.fig.suptitle("Batches by State and Season")
            for ax in g.axes.ravel():
                # add annotations
                for c in ax.containers:
                    # labels = [f'{(v.get_width() / 1000):.1f}K' for v in c]
                    labels = [int(v.get_width()) for v in c]
                    ax.bar_label(c, labels=labels, label_type="edge", fontsize=14)
                ax.margins(y=0.2)
            g.savefig(
                f"{self.figs_results_dir}/batches_by_state_and_season.png",
                dpi=150,
            )

    def totals(self):
        totals = [
            [
                self.batch_total,
                self.image_total,
                self.cutout_total,
                self.total_primary_cutouts,
                self.total_species,
            ]
        ]
        df = pd.DataFrame(
            totals,
            columns=[
                "total_batches",
                "total_images",
                "total_cutouts",
                "total_primary_cutouts",
                "total_species",
            ],
        )

        tdf = df.T
        tdf.to_csv(f"{self.metrics_results_dir}/totals.csv", header=False)


# def main(cfg: DictConfig) -> None:
#     # Example usage
#     processor = ParquetDataProcessor(cfg)
#     processor.read_parquet_file()
#     # # processor.table_for_pub()
#     processor.cutouts_by_common_name_state_and_season()
#     processor.image_by_common_name()
#     processor.images_by_common_name_state(save_fig=True)
#     processor.images_by_species_and_season(save_fig=True)
#     processor.cutouts_by_common_name()
#     processor.cutouts_by_common_name_state(save_fig=True)
#     processor.primary_cutouts_by_common_name()
#     processor.primary_status_cutouts_by_common_name_state(save_fig=True)
#     processor.primary_cutouts_by_common_name_general_season(save_fig=True)
#     processor.primary_weed_cutouts_by_common_nameEPPO_state(save_fig=True)

#     processor.cutouts_by_species_and_season(save_fig=True)
#     processor.unique_batches_by_state_season(save_fig=True)
#     processor.totals()

#     # sampler = SampleImageData(cfg, processor.df)
#     # sampler.copy_cutouts()
