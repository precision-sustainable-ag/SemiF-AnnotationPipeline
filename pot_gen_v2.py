from ast import Break
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from semif_utils.datasets import Background, Cutout, Pot
from semif_utils.mongo_utils import Connect

"""

Get directories

Inputs: 
1. cutouts
2. pots
3. empty backgrounds

Outputs:
1. synth images
2. synth masks
"""

class ValidateDirs:
    def __init__(self, cfg: DictConfig) -> None:
        """Checks if input and output directories exists, creates output directories if necessary,
           and creates output directory paths for processing.

        Args:
            cfg (DictConfig): synth_config.yaml

        Raises:
            [directory] Directory Not Found: if the directory does not exist and needs to be created

        Returns:
            imgdir (str): path to output directory for images
            maskdir (str): path to output directory for masks
        """
        self.datadir = cfg.general.datadir
  
## get and validate directories

# get Inputs
# validate and/or create

# get Outputs
# validate and/or create

## Create data container for cutouts, pots, and backgrounds 

@dataclass
class SynthDataContainer:
    """Combines documents in a database with items in a directory to form data container for generating synthetic bench images. Includes lists of dataclasses for cutouts, pots, and backgrounds.
    
    Args:
        datadir (Path): Parent directory that should contain "cutouts", "pots", and "backgrounds".
        cutouts list[Cutouts]: list of Cutout dataclasses
        pots list[Pot]: list of Pot dataclasses
        backs list[Background]: list of Background dataclasses

    Returns:
        data_container (object): Dataclass with Cutouts, Pots, and Backgrounds itemized in a mongoDB.
    """
    datadir: Path
    db_attr: str = "test"
    to_db: bool = False
    cutouts: list[Cutout] = field(init=False, default=None)
    pots: list[Pot] = field(init=False, default=None)
    backs: list[Background] = field(init=False, default=None)

    def __post_init__(self):
        if self.to_db:
            self.from_db()
        else:
            self.from_dir()
    
    def from_dir(self):
        self.cutouts =  self.datadir.glob("cutouts/*.png")
        self.pots =     self.datadir.glob("pots/*.png")
        self.backs =    self.datadir.glob("backgrounds/*.png")


    def from_db(self):
        self.backs = self.connect_db_dir_products("Backgrounds")
        self.pots = self.connect_db_dir_products("Pots")
        self.cutouts = self.connect_db_dir_products("Cutouts")


    def query_db(self, db_collection):
        connection = Connect.get_connection()
        db = getattr(connection, self.db_attr)
        cursor = getattr(db, db_collection).find()
        return cursor


    def connect_db_dir_products(self, collection_str):
        """Connnects documents in a database collection with items in a directory.
        Places connected items in a list of dataclasses.
        """             
        syn_datacls = {"cutout": Cutout, "pot": Pot, "background": Background} 
        cursor = self.query_db(collection_str.title())
        docdir = Path(self.datadir,collection_str.lower())
        path_str_ws = collection_str.lower().replace("s", "")
        path_str = f"{path_str_ws}_path"
        docs = []
        for doc in cursor:
            doc_path = docdir /doc[path_str]
            doc[path_str] = str(doc_path)
            assert Path(doc_path).exists(), f"Image with path {str(doc_path)} does not exist."
            doc.pop("_id")
            data_cls = syn_datacls[path_str_ws]
            docs.append(data_cls(**doc))
        return docs


@dataclass
class PottedBackground:
    background_id: str
    background_path: str
    pot_ids: list
    pot_paths: list
    pot_positions: list
    arr: np.ndarray
    mask: np.ndarray    

datadir = Path("data/synth_data_test")
syndata = SynthDataContainer(datadir=datadir)

numimgs = 20

for img in syndata.cutouts:
    im = cv2.imread(str(img))
    print(type(im))    
    print(im.shape)
    print(im.dtype)

    
## Overlay pot on background
class OverlayPotOnBackground:
    """
    Inputs: 
        background image: 
        list of Pots    : list of pot dataclasses for overlaying on background

    Methods:
        get_pots_for_background (# of pots): (list of data classes) list of pot dataclasses (selection choices include; random, selective, or weighted

        transform_pot  (image) : (image)  add transformation (rotateion, flip, scale, change brightness, etc.). Update basic pot image data in dataclass

        get_pot_center_positions (bkgd image) : a dict like, "pos1":(x,y). Get positions on background of where to place pot centers

        adjust_positions : adjust pot position based on background image size and pot size. Avoid pot completely being out of frame. Should compare x,y positions and h,w background image and h,w pot image information. Update pot position data in pot dataclass

        create_mask : this mask won't be saved but is used for pasting pots on background

        paste_foreground : use mask to paste pots on background. 

    Returns:
        PottedBackground dataclass with image, mask, pot positions, and other info.
    """
    def __init__(self):
        self.pots = False
        self.background = False

    def get_pots(num_pots):
        pass
    
    def transform_pot(self):
        pass


@dataclass
class SynthData:
    pass

## Overlay plant on pot

""" 
Inputs:
    PottedBackground 
    
    Pot positions (dict)    :   pot_id:[pot_position]
    back_w_pots (np.array)  :   background image with overlain pots
    cutouts list[Cutout]    :   List of Cutout dataclasses representing images 
                                to be place on that background with pots

Methods:
    cutout_list list[Cutout]:   Gets list of cutouts for each back_w_pots images.
"""

