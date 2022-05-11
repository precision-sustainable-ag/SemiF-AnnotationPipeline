import random
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torchvision.ops.boxes as bops

from semif_utils.datasets import Background, Cutout, Pot, TempCutout
from semif_utils.mongo_utils import Connect


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
    datadir: str
    db_db: str = None
    use_db: bool = True
    cutouts: list[Cutout] = field(init=False, default=None)
    pots: list[Pot] = field(init=False, default=None)
    backgrounds: list[Background] = field(init=False, default=None)

    def __post_init__(self):
        if self.use_db:
            self.from_db()
        else:
            self.from_dir()

    def from_dir(self):

        self.cutouts = Path(self.datadir).glob("cutouts/*.png")
        self.pots = Path(self.datadir).glob("pots/*.png")
        self.backgrounds = Path(self.datadir).glob("backgrounds/*.png")

    def from_db(self):
        self.backgrounds = self.connect_db_dir_products("Backgrounds")
        self.pots = self.connect_db_dir_products("Pots")
        self.cutouts = self.connect_db_dir_products("Cutouts")

    def query_db(self, db_collection):
        connection = Connect.get_connection()
        db = getattr(connection, self.db_db)
        cursor = getattr(db, db_collection).find()
        return cursor

    def connect_db_dir_products(self, collection_str):
        """Connnects documents in a database collection with items in a directory.
        Places connected items in a list of dataclasses.
        """
        syn_datacls = {
            "cutout": Cutout,
            "pot": Pot,
            "background": Background
        }
        cursor = self.query_db(collection_str.title())
        docdir = Path(self.datadir, collection_str.lower())
        path_str_ws = collection_str.lower().replace("s", "")
        path_str = f"{path_str_ws}_path"
        docs = []
        for doc in cursor:
            doc_path = doc[path_str]  #docdir / doc[path_str]
            doc[path_str] = str(doc_path)
            assert Path(doc_path).exists(
            ), f"Image with path {str(doc_path)} does not exist."
            doc.pop("_id")
            data_cls = syn_datacls[path_str_ws]
            pprint(doc)
            docs.append(data_cls(**doc))
        return docs

def bbox_iou(box1, box2):
    box1 = torch.tensor([box1], dtype=torch.float)
    box2 = torch.tensor([box2], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou      

def get_img_bbox(x, y, imgshape):
    pot_h, pot_w, _ = imgshape
    x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
    bbox = [x0, y0, x1, y1] # top right corner, bottom left corner
    return bbox

def center2topleft(x, y, background_imgshape):
    """ Gets top left coordinates of an image from center point coordinate
    """
    back_h, back_w, _ = background_imgshape
    y = y - int(back_h/2)
    x = x - int(back_w/2)
    return x, y

def transform_position(points, imgshape, spread_factor):
    """ Applies random jitter factor to points and transforms them to top left image coordinates. 
    """
    y, x = points

    x, y = x + random.randint(-spread_factor, spread_factor), y + random.randint(-int(spread_factor/3), int(spread_factor/3))
    
    x, y = center2topleft(x, y, imgshape)
        
    return x, y

def center_on_background(y, x, back_shape, fore_shape):
    # pot positions and shape top left corner
    back_h, back_w, _ = back_shape
    fore_h, fore_w, _ = fore_shape
    newx = int(((back_w - fore_w) / 2) + x)
    newy = int(((back_h - fore_h) / 2) + y)
    return newx, newy


def img2RGBA(img):
    alpha = np.sum(img, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    img = np.dstack((img, alpha))
    return img

# Pot positioning
sixpot = {
    0:(1592, 1599),
    1:(1592, 4796),
    2:(1592, 7993),
    3:(4776, 1599),
    4:(4776, 4796),
    5:(4776, 7993)
}

POTMAPS = [sixpot]
