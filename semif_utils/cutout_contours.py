from collections import defaultdict

import cv2
import Metashape
import numpy as np
from shapely.geometry import MultiPolygon, Polygon


class CutoutMapper(object):
    """ Adapted from bbox/bbox_transformations.py"""
    def __init__(self, project_path: str):
        """Class to map bounding box coordinates from image cordinates
           to global coordinates
        """
        self.doc = Metashape.Document()
        self.doc.open(str(project_path))

    def map(self, cutouts, image_id):
        """
        Maps all the bounding boxes to a global coordinate space
        """
        global_coordinates = []
        for cutout in cutouts:

            # Isolate the chunk
            camera_chunk = None
            for chunk in self.doc.chunks:
                cameras = [camera.label for camera in chunk.cameras]
                if image_id in cameras:
                    camera_chunk = chunk
                    camera = [cam for cam in chunk.cameras if cam.label == image_id][0]
                    break
            
            assert camera_chunk is not None

            # From: https://www.agisoft.com/forum/index.php?topic=13875.0
            surface = camera_chunk.point_cloud
            mapped_coordinates = []
            dup = []
            for x_coord, y_coord in cutout:                
                # for x_coord, y_coord in contour:
                ray_origin = camera.center # camera.unproject(Metashape.Vector([x_coord, y_coord, 0]))
                ray_target = camera.unproject(Metashape.Vector([x_coord, y_coord]))

                point_internal = surface.pickPoint(ray_origin, ray_target)

                if point_internal is None:
                    raise TypeError()
                
                # From https://www.agisoft.com/forum/index.php?topic=12781.0
                global_coord = camera_chunk.crs.project(camera_chunk.transform.matrix.mulp(point_internal))[:2]
                global_coord = list(global_coord)
                if global_coord == dup:
                    continue
                mapped_coordinates.append(global_coord)
                dup = global_coord
            global_coordinates.append(mapped_coordinates)
        return global_coordinates


def mask_to_polygons(mask, epsilon=10., min_area=10., to_list=True):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # https://michhar.github.io/masks_to_polygons_and_back/
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask,
                                  cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # all_polygons = MultiPolygon(all_polygons)
    if to_list:
        all_polygons = shapely_poly_to_list(all_polygons)
    return all_polygons

def polygons_to_mask(polygons, im_size):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def shapely_poly_to_list(polygons):
    poly_lists = []
    for poly in polygons:
        xs = list(poly.exterior.coords.xy[0])
        ys = list(poly.exterior.coords.xy[1])
        xy = [[int(x),int(y)] for x, y in zip(xs,ys)]
        poly_lists.append(xy)
    return poly_lists

# Testing
# masks = [x for x in Path("../SemiF-SyntheticPipeline/data/semifield-synth/masks_altered_v2").glob("*.png")]

# idx = 1
# mask = cv2.imread(str(masks[idx]), cv2.IMREAD_UNCHANGED)
# from pprint import pprint

# # Get the polygons using shapely
# polys = mask_to_polygons(mask, min_area=10, to_list=True)
# CutoutMapper()
    # xs = [x[0] for x in poly]
    # pprint(xs)
# pprint(polys)
# cvt_mask = polygons_to_mask(polys, mask.shape)

