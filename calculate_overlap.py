import Metashape
import os
import itertools
import math
import multiprocessing
import concurrent.futures
from shapely.geometry import Polygon
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def create_footprints():
    """
    Creates four-vertex shape for each aligned camera (footprint) in the active chunk
    and puts all these shapes to a new separate shape layer
    """

    doc = Metashape.Document()
    # project_path = Path("/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images/MD_2024-07-08/autosfm_good/project/MD_2024-07-08.psx")
    # project_path = Path("/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images/MD_2022-06-27/autosfm/project/MD_2022-06-27.psx")
    # project_path = Path("/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images/MD_2024-07-03/autosfm/project/MD_2024-07-03.psx")
    project_path = Path("/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images/NC_2024-07-15/autosfm_df1_apd2_744/project/NC_2024-07-15.psx")
    doc.open(str(project_path), read_only=False, ignore_lock=True)
    if not len(doc.chunks):
        raise Exception("No chunks!")

    print("Script started...")
    chunk = doc.chunk

    if not chunk.shapes:
        chunk.shapes = Metashape.Shapes()
        chunk.shapes.crs = chunk.crs
    T = chunk.transform.matrix
    footprints = chunk.shapes.addGroup()
    footprints.label = "Footprints"
    footprints.color = (30, 239, 30)

    if chunk.model:
        surface = chunk.model
    elif chunk.point_cloud:
        surface = chunk.point_cloud
    else:
        surface = chunk.tie_points
        
    camera_footprints = {}
    def process_camera(chunk, camera):
        if camera.type != Metashape.Camera.Type.Regular or not camera.transform:
            return  # skipping NA cameras

        sensor = camera.sensor
        corners = list()
        for (x, y) in [[0, 0], [sensor.width - 1, 0], [sensor.width - 1, sensor.height - 1], [0, sensor.height - 1]]:
            ray_origin = camera.unproject(Metashape.Vector([x, y, 0]))
            ray_target = camera.unproject(Metashape.Vector([x, y, 1]))
            corners.append(surface.pickPoint(ray_origin, ray_target))
            if not corners[-1]:
                corners[-1] = chunk.tie_points.pickPoint(ray_origin, ray_target)
            if not corners[-1]:
                break
            corners[-1] = chunk.crs.project(T.mulp(corners[-1]))

        if not all(corners):
            print("Skipping camera " + camera.label)
            return

        if len(corners) == 4:
            shape = chunk.shapes.addShape()
            shape.label = camera.label
            shape.attributes["Photo"] = camera.label
            shape.group = footprints
            shape.geometry = Metashape.Geometry.Polygon(corners)
            camera_footprints[camera.label] = Polygon([(p.x, p.y) for p in corners])

    max_workers = int(len(os.sched_getaffinity(0)) / 5)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map(lambda camera: process_camera(chunk, camera), chunk.cameras)
    
    for camera in tqdm(chunk.cameras):
        process_camera(chunk, camera)
    
    # Calculate average overlap
    total_percentage_overlap = 0
    num_pairs = 0
    images_with_no_overlap = set(camera_footprints.keys())
    overlap_counts = defaultdict(int)

    with open(project_path.stem + ".txt", "w") as file:
        camera_labels = list(camera_footprints.keys())
        for i in range(len(camera_labels)):
            for j in range(i + 1, len(camera_labels)):
                poly1 = camera_footprints[camera_labels[i]]
                poly2 = camera_footprints[camera_labels[j]]
                if poly1.intersects(poly2):
                    try:
                        intersection_area = poly1.intersection(poly2).area
                    except Exception as e:
                        intersection_area = 0
                    try:
                        union_area = poly1.union(poly2).area
                        percentage_overlap = (intersection_area / union_area) * 100
                    except Exception as e:
                        percentage_overlap = 0

                    total_percentage_overlap += percentage_overlap
                    num_pairs += 1
                    file.write(f"{camera_labels[i]} and {camera_labels[j]}: {percentage_overlap:.2f}%\n")
                    images_with_no_overlap.discard(camera_labels[i])
                    images_with_no_overlap.discard(camera_labels[j])
                    # Count overlaps
                    overlap_counts[camera_labels[i]] += 1
                    overlap_counts[camera_labels[j]] += 1

        if num_pairs > 0:
            average_percentage_overlap = total_percentage_overlap / num_pairs
        else:
            average_percentage_overlap = 0

        file.write(f"\nAverage percentage overlap: {average_percentage_overlap:.2f}%\n")
        file.write(f"\nImages with no overlap ({len(images_with_no_overlap)}):\n")
        for img in images_with_no_overlap:
            file.write(f"{img}\n")

        file.write("\nNumber of overlapping images per image:\n")
        total_overlaps = 0
        overlapping_images = 0
        for img in sorted(camera_labels):
            overlaps = overlap_counts[img]
            total_overlaps += overlaps
            if overlaps > 0:
                overlapping_images += 1
            file.write(f"{img}: {overlaps}\n")

        if overlapping_images > 0:
            average_overlapping_images = total_overlaps / overlapping_images
        else:
            average_overlapping_images = 0

        file.write(f"\nAverage number of overlapping images per image: {average_overlapping_images:.2f}\n")
        

    print(f"Average percentage overlap: {average_percentage_overlap:.2f}%")
    print("Script finished!")

create_footprints()
