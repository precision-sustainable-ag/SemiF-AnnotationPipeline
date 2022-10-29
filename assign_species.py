import json
import logging
from pathlib import Path

import fiona
import geopandas
import matplotlib.path as mplPath
import numpy as np
from dacite import Config, from_dict
from omegaconf import DictConfig
from shapely.geometry import MultiPolygon, Point, shape
from tqdm import tqdm

from semif_utils.datasets import ImageData
from semif_utils.segment_utils import get_species_info, load_speciesinfo

log = logging.getLogger(__name__)


def inpolygon(point, polygon):
    bb_path = mplPath.Path(np.array(polygon))
    return bb_path.contains_point(point)


def main(cfg: DictConfig) -> None:
    """ Identifies species by comparing shapefile polygon and global bbox centroid point.
    """
    # Config variables
    batch_id = cfg.general.batch_id
    species_info = cfg.data.species
    location = batch_id.split("_")[0]
    metadata_path = Path(cfg.data.batchdir, "metadata")
    spec_dict = load_speciesinfo(species_info)
    # Get Json metadata files
    image_metadata_files = sorted(list(metadata_path.glob("*.json")))

    # Load shp file twice
    shapefile_path = Path(cfg.data.utilsdir, location, "shapefiles",
                          f"{location}.shp")
    # Using Geopandas for poly contains point
    polys = geopandas.GeoDataFrame.from_file(shapefile_path)
    # Using fioana for check if point is in poly
    Multi = MultiPolygon(
        [shape(pol["geometry"]) for pol in fiona.open(shapefile_path)])

    # Iterate over json files
    for file in tqdm(image_metadata_files, desc="Assigning labels"):
        # Read metadata and create ImageData dataclass
        with open(file) as f:
            j = json.load(f)
            imgdata = from_dict(data_class=ImageData,
                                data=j,
                                config=Config(check_types=False))
        # Iterate over image bboxes
        for bbox in imgdata.bboxes:
            # print(bbox.cls)
            x = bbox.global_centroid[0]
            y = bbox.global_centroid[1]
            bbox_cls = bbox.cls
            point = Point(x, y)
            # Identify shp file polygon that contains bbox centroid point
            for poly in polys.itertuples():
                if bbox_cls == "colorchecker":
                    spec_info = spec_dict["species"][bbox_cls]
                    break
                # Check if species polygon contains bbox centroid
                if poly.geometry.contains(point):
                    poly_cls = poly.species
                    if poly_cls in spec_dict["species"].keys():
                        spec_info = spec_dict["species"][poly_cls]
                        break
                if not point.within(Multi):
                    # If centroid lies outside of any polygon
                    spec_info = spec_dict["species"]["plant"]
                    log.info(
                        f"{bbox.bbox_id} centroid not within any potting group polygon for image {imgdata.image_id}. Assigning default class 'plant'"
                    )
                    break
            bbox.assign_species(spec_info)
        # Save
        imgdata.save_config(metadata_path)
    log.info(f"Assigning species complete for batch {batch_id}")
