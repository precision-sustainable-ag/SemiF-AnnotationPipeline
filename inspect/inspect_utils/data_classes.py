from pathlib import Path
from dataclasses import dataclass, field
import pandas
import logging

log = logging.getLogger(__name__)


@dataclass
class InspectionSamples:

    df: pandas.DataFrame
    sample_size: int
    temp_cropout_parent_dir: str

    md_sample: pandas.DataFrame = field(init=False)
    nc_sample: pandas.DataFrame = field(init=False)

    # TODO tx_paths: list = field(init=False)

    def __post_init__(self):
        self.state_dict = self.data_dict()
        # MD
        self.md_sample = self.get_sample("MD")

        # NC
        self.nc_sample = self.get_sample("NC")

        # TX
        # self.tx_sample = self.get_sample("TX")

    def data_dict(self):
        state_dict = {
            "MD": {
                "batches": list(),
                "species": list(),
                "df": pandas.DataFrame()
            },
            "NC": {
                "batches": list(),
                "species": list(),
                "df": pandas.DataFrame()
            },
            # "TX": {
            #     "batches": list(),
            #     "species": list(),
            #     "df": pandas.DataFrame()
            # }
        }
        return state_dict

    def get_state_lists(self, state):
        # state_dict = self.data_dict()
        state_df = self.df[self.df["state_id"] == state]
        batches = state_df["batch_id"].unique()
        species = state_df["common_name"].unique()

        self.state_dict[state].update({"batches": batches})
        self.state_dict[state].update({"species": species})
        self.state_dict[state].update({"df": state_df})

        return self.state_dict

    def set_temp_paths(self, state):
        state_df = self.df[self.df["state_id"] == state]
        stdfc = state_df.copy()

        stdfc[
            "temp_cropout_path"] = self.temp_cropout_parent_dir + "/" + state_df[
                "cutout_path"].str.replace("png", "jpg")
        # Add additional column to the Beginning
        stdfc = stdfc.reindex(columns=["true_species"] +
                              stdfc.columns.tolist())

        return stdfc

    def get_sample(self, state):
        self.get_state_lists(state)
        batches = self.state_dict[state]["batches"]
        species = self.state_dict[state]["species"]
        df = self.set_temp_paths(state)

        samples = []
        for batch in batches:
            batch_df = df[df["batch_id"] == batch]

            for spec in species:
                batch_copy = batch_df.copy()
                spec_df = batch_copy[batch_copy["common_name"] == spec]

                if state == "NC":
                    tspec_df = spec_df.copy()
                    spec_df = tspec_df[tspec_df["green_sum"] > 1500]

                num_rows = spec_df.shape[0]
                if num_rows > 0:
                    if num_rows > self.sample_size:
                        samp_size = self.sample_size
                    else:
                        samp_size = num_rows
                    samp = spec_df.sample(n=samp_size, random_state=42)
                    samples.append(samp)
                # else:
                # log.warning(f"No {spec} for {batch}")

        final_state_df = pandas.concat(samples).reset_index(drop=True).iloc[:,
                                                                            1:]
        return final_state_df
