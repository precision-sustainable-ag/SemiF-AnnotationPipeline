import random
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

from semif_utils.datasets import Background, Pot, TempCutout
from semif_utils.mongo_utils import Connect, from_db, to_db
from synth_utils_v2 import SynthDataContainer


## Pipeline start
class GenPottedBackgrounds:

    def __init__(self, datacontainer, cfg: DictConfig) -> None:
        self.synth_config = cfg.synth
        self.count = cfg.synth.count
        self.back_config = cfg.synth.backgrounds
        self.pot_config = cfg.synth.pots

        self.backgrounds = datacontainer.backgrounds
        self.used_backs = []

        self.pots = datacontainer.pots
        # TODO make sure number of images does does not exceed needed number of pots
        self.used_pots = []

        self.cutouts = datacontainer.cutouts

        self.with_replace = False if len(
            self.backgrounds) > self.count else True

    def get_backs(self, sortby=None):
        if not self.with_replace:
            self.backgrounds = [
                x for x in self.backgrounds if x not in self.used_backs
            ]
        if self.back_config.selection == "random":
            backs = random.choice(self.backgrounds)
        return backs

    def get_num_pots(self):
        num = random.int(len(list(self.pots)))

    def get_pots(self, num_pots, sortby=None):
        if self.pot_config.selection == "random":
            pots = [x for x in self.pots if x not in self.used_pots]
            self.used_pots.append(pots)

            num_pots = num_pots if num_pots < len(pots) else len(pots)

            pots = random.sample(pots, num_pots)
        return pots

    def get_pot_positions(self):
        pass

    def pot_center(self, pot_shape):
        cent_h = pot_shape[0] - int(pot_shape[0] / 2)
        cent_w = pot_shape[1] - int(pot_shape[1] / 2)
        return cent_h, cent_w

    def overlay_pots(self, back, pots):
        background = back.array.copy()
        back_h, back_w, back_c = background.shape

        for pot in pots:
            pot_rgba = pot.array.copy()
            pot_h, pot_w = self.pot_center(self, pot_rgba.shape[:2])
            pot_mask = pot_rgba[..., 3:] / 255
            alpha_l = 1.0 - pot_mask

            background[300:740, 600:1033] = alpha_l * background[
                300:740, 600:1033] + pot_mask * pot

            # TODO add transformations here
            # TODO Get pot positions here

    @property
    def pipeline(self):
        for imgnum in range(self.count):
            back = self.get_backs()
            self.used_backs.append(back)
            # Start pipeline
            pots = self.get_pots(3)  # TODO make this number dyanmic/random
            self.overlay_pots(back, pots)


def main(cfg: DictConfig) -> None:
    # Create cutout select (by batch, location, time, size, etc.)
    # Get data from database
    syn = SynthDataContainer(datadir=cfg.synth.datadir, db_db="test")
    potback = GenPottedBackgrounds(syn, cfg).pipeline

    # print(syn)
    # # Parse data
    # for back in syn.backgrounds:
    #     h, w, c = back.array.shape
    #     print(h)
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg)
