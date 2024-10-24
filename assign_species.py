import json
from pathlib import Path

import matplotlib.path as mplPath
import numpy as np
import shapefile
from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from semif_utils.datasets import ImageData
from semif_utils.segment_utils import get_species_info


def inpolygon(point, polygon):
    bb_path = mplPath.Path(np.array(polygon))
    return bb_path.contains_point(point)


def main(cfg: DictConfig) -> None:

    batch_id = cfg.general.batch_id
    species_info = cfg.data.species
    location = batch_id.split("_")[0]
    shapefile_path = Path(cfg.data.utilsdir, location, "shapefiles",
                          f"{location}.shp")

    # Read the shapefiles
    sf = shapefile.Reader(shapefile_path)
    polygons = sf.shapeRecords()

    metadata_path = Path(cfg.data.batchdir, "metadata")
    image_metadata_files = metadata_path.glob("*.json")

    for file in tqdm(image_metadata_files, desc="Assigning labels"):
        with open(file) as f:
            j = json.load(f)
            imgdata = from_dict(data_class=ImageData,
                                data=j,
                                config=Config(check_types=False))
        for bbox in imgdata.bboxes:
            # Assign species
            for polygon in polygons:

                # polygon_bbox is structured as [x_lower,y_lower,x_upper,y_upper]
                polygon_bbox = polygon.shape.bbox
                # Find if significant overlap
                horizontal = polygon_bbox[0] < bbox.global_centroid[
                    0] and polygon_bbox[2] > bbox.global_centroid[0]
                vertical = polygon_bbox[1] < bbox.global_centroid[
                    1] and polygon_bbox[3] > bbox.global_centroid[1]

                in_polygon_bbox = horizontal and vertical

                if in_polygon_bbox:
                    # Find if in the polygon
                    # Fix handling polygon with inner parts
                    parts_in_polygon = polygon.shape.parts

                    if len(parts_in_polygon) == 1:
                        polygon_points = polygon.shape.points
                    else:
                        polygon_points = polygon.shape.points[
                            0:parts_in_polygon[1]]

                    x, y = bbox.global_centroid[0], bbox.global_centroid[1]
                    in_polygon = inpolygon((x, y), polygon_points)

                    if in_polygon:
                        spec_info = get_species_info(species_info,
                                                     polygon.record["species"])
                        bbox.assign_species(spec_info)
                        # get_species_info(species_info)

                        break
        # Save
        imgdata.save_config(metadata_path)

    # Close to avoid memory leak
    sf.close()
