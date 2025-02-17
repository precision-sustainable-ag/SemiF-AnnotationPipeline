import json
import logging
import time
from pathlib import Path

import geopandas
import matplotlib.path as mplPath
import numpy as np
from dacite.config import Config
from dacite.core import from_dict
from omegaconf import DictConfig
from shapely.geometry import Point
from tqdm import tqdm

from semif_utils.datasets import ImageData
from semif_utils.segment_utils import load_speciesinfo

log = logging.getLogger(__name__)


def inpolygon(point, polygon):
    bb_path = mplPath.Path(np.array(polygon))
    return bb_path.contains_point(point)


def main(cfg: DictConfig) -> None:
    """Identifies species by comparing shapefile polygon and global bbox centroid point."""
    start = time.time()
    # Config variables
    species_info = cfg.data.species
    metadata_path = Path(cfg.data.batchdir, "metadata")
    spec_dict = load_speciesinfo(species_info)
    # Get Json metadata files
    image_metadata_files = sorted(list(metadata_path.glob("*.json")))

    # Load shp file twice
    shapefile_path = Path(cfg.data.species_poly)
    # Using Geopandas for poly contains point
    polygons_filtered = geopandas.read_file(shapefile_path)
    # polygons_filtered = polys.dropna(subset=["species"])

    # Iterate over json files
    for file in tqdm(image_metadata_files, desc="Assigning labels"):
        # Read metadata and create ImageData dataclass
        with open(file) as f:
            j = json.load(f)
            imgdata = from_dict(
                data_class=ImageData, data=j, config=Config(check_types=False)
            )
        batch_id = imgdata.batch_id
        # Iterate over image bboxes
        for bbox in imgdata.bboxes:
            # print(bbox.cls)
            x = bbox.global_centroid[0]
            y = bbox.global_centroid[1]
            bbox_cls = bbox.cls
            if bbox_cls == "colorchecker":
                # spec_info = spec_dict["species"][bbox_cls]
                poly_cls = bbox_cls

            elif "cash" in cfg.general.season and bbox_cls != "colorchecker":
                if "NC" in cfg.general.batch_id:
                    poly_cls = "GLMA4"
                elif "MD" in cfg.general.batch_id:
                    poly_cls = "ZEA"
                elif "TX" in cfg.general.batch_id:
                    poly_cls = "GOHI"
            else:
                point = Point(x, y)

                # Get polygons that contain point
                contains_point = polygons_filtered["geometry"].apply(
                    lambda polygon: polygon.contains(point)
                )
                containing_polygon = polygons_filtered[contains_point]

                if len(containing_polygon) == 0:
                    log.warning(
                        f"Global centroid was not found in any polygons for bbox_id: {bbox.bbox_id}. Finding the closest polygon less than 1 meter away."
                    )

                    # Get the distances to each polygon
                    distances = polygons_filtered["geometry"].apply(
                        lambda polygon: polygon.distance(point)
                    )

                    # Find the closest distance
                    closest_polygon_index = distances.idxmin()
                    closest_distance = distances[closest_polygon_index]
                    closest_distance_thresh = 2
                    if closest_distance < closest_distance_thresh:
                        closest_polygon = polygons_filtered.loc[closest_polygon_index]
                        poly_cls = closest_polygon["species"]
                        log.warning(
                            f"Global centroid found in the next closest polygon less than {closest_distance_thresh} meter away: {closest_polygon['comm_name']}"
                        )

                    else:
                        poly_cls = None
                        log.warning(
                            f"No polygon or closest polygon (< {closest_distance_thresh} meter) was found. Exiting.{closest_distance}\n{poly_cls}\n{file}\n{point}"
                        )
                        log.info(f"Point: {point}")
                        
                        
                else:
                    poly_cls = containing_polygon["species"].values[0]
                    poly_state = containing_polygon["state"].values[0]
                    poly_becnh = containing_polygon["bench"].values[0]
                    poly_comm_name = containing_polygon["comm_name"].values[0]
                    poly_class_id = containing_polygon["class_id"].values[0]
                    poly_state = poly_state if poly_state is not None else None
                    poly_becnh = poly_becnh if not np.isnan(poly_becnh) else None
                    poly_comm_name = poly_comm_name if poly_comm_name is not None else None
                    poly_class_id = poly_class_id if poly_class_id is not None else None

                    if poly_state is None and poly_becnh is None and poly_comm_name is None and poly_class_id is None:
                        log.warning(
                            f"Global centroid was found in a polygon but the species is not defined. Labeling as non_target_weed."
                        )
                        
                        poly_id = containing_polygon["id"].values[0]
                        if batch_id == "MD_2023-04-25":
                            if int(poly_id) == 20:
                                poly_cls = "VIVI"
                            else:
                                poly_cls = None
                                if bbox.non_target_weed != 'non_target_weed':
                                    bbox.non_target_weed = 'non_target_weed'
                                    bbox.non_target_weed_pred_conf = 1.0
                        else:
                            poly_cls = None    
                            if bbox.non_target_weed != 'non_target_weed':
                                bbox.non_target_weed = 'non_target_weed'
                                bbox.non_target_weed_pred_conf = 1.0
            
            if poly_cls is None:
                spec_info = spec_dict["species"]['plant']
            else:
                spec_info = spec_dict["species"][poly_cls]

            bbox.assign_species(spec_info)
        # Save
        imgdata.save_config(metadata_path)

    end = time.time()
    log.info(f"Assigning species completed in {end - start} seconds.")
