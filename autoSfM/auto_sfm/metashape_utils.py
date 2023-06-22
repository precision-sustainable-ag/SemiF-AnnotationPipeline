import glob
import logging
import os
from collections import Counter
from copy import deepcopy
from typing import Callable

import cv2
import Metashape as ms
import numpy as np
import rasterio as rio
from PIL import Image

from .callbacks import percentage_callback
from .dataframe import DataFrame
from .estimation import (CameraStats, MarkerStats, field_of_view,
                         find_object_dimension)

log = logging.getLogger(__name__)


class SfM:

    def __init__(self, cfg):

        self.cfg = cfg
        self.metashape_key = cfg.general.metashape_key
        self.doc = self.load_or_create_project(cfg["asfm"]["proj_path"])
        self.metashape_key = cfg.general.metashape_key
        self.num_gpus = cfg["asfm"]["num_gpus"] if cfg["asfm"][
            "num_gpus"] != "all" else 2**len(ms.app.enumGPUDevices()) - 1

    def load_or_create_project(self, project_path: str) -> ms.Document:
        """Opens a project if it exists or creates and saves a project

        Returns:
            ms.Document: Metashape Document containing the project
        """
        project_path = self.cfg["asfm"]["proj_path"]
        assert project_path.split(".")[-1] == "psx"
        if not ms.app.activated:
            ms.License().activate(self.metashape_key)
        doc = ms.Document()
        batch_id = self.cfg["general"]["batch_id"]
        if os.path.exists(project_path):
            log.info(
                f"Metashape project already exisits. Opening project file.")
            # Metashape window
            doc.open(project_path, read_only=False, ignore_lock=True)
        else:
            # Create a project and save
            log.info(f"Creating new Metashape project for {batch_id}")
            doc.addChunk()
            doc.save(project_path)

        return doc

    def save_project(self):
        """Save the project
        """
        project_path = self.cfg["asfm"]["proj_path"]
        assert project_path.split(".")[-1] == "psx"
        self.doc.save(project_path)

    def add_photos(self):
        """Adds a directory to the project
        """
        photos = glob.glob(
            os.path.join(self.cfg["asfm"]["down_photos"], "*.jpg"))

        if self.doc.chunk is None:
            self.doc.addChunk()
        self.doc.chunk.addPhotos(photos)

    def add_masks(self):
        """Adds masks to the cameras
        """
        self.doc.chunk.generateMasks(
            path=self.cfg["asfm"]["down_masks"] + "/{filename}_mask.png",
            masking_mode=ms.MaskingMode.MaskingModeFile,
            cameras=self.doc.chunk.cameras)
        self.save_project()

    def detect_markers(self,
                       chunk: int = 0,
                       progress_callback: Callable = percentage_callback):
        """Detects 12 bit circular markers
        """
        self.doc.chunks[chunk].detectMarkers(
            target_type=ms.CircularTarget12bit,
            tolerance=50,
            filter_mask=False,
            inverted=False,
            noparity=False,
            maximum_residual=5,
            minimum_size=0,
            minimum_dist=5,
            cameras=self.doc.chunks[chunk].cameras,
            progress=progress_callback)
        self.save_project()

    def import_reference(self, chunk: int = 0):
        """Imports reference points
        """
        self.doc.chunks[chunk].importReference(
            path=self.cfg["data"]["gcp_ref"],
            format=ms.ReferenceFormatCSV,
            columns="[n|x|y|z]",
            delimiter=";",
            group_delimiters=False,
            skip_rows=1,
            ignore_labels=False,
            create_markers=True,
            threshold=0.1,
            shutter_lag=0)
        self.save_project()

    def export_camera_reference(self):
        """Exports the reference to a CSV file.
        """
        # self.doc.chunk = self.doc.chunks[1]
        reference = []
        for camera in self.doc.chunk.cameras:
            stats = CameraStats(camera).to_dict()
            calibration_params = self.camera_paramters(camera)
            stats.update(calibration_params)
            # Check camera alignment
            # https://www.agisoft.com/forum/index.php?topic=6029.msg29219#msg29219
            is_aligned = camera.transform is not None
            stats.update({"Alignment": is_aligned})
            reference.append(stats)

        dataframe = DataFrame(reference, "label")
        self.camera_reference = dataframe
        dataframe.to_csv(self.cfg["asfm"]["cam_ref"], header=True, index=False)

    def export_gcp_reference(self):
        reference = []
        for marker in self.doc.chunk.markers:
            stats = MarkerStats(marker).to_dict()
            is_detected = len(marker.projections.items()) > 0
            stats.update({"Detected": is_detected})
            reference.append(stats)

        dataframe = DataFrame(reference, "label")
        self.gcp_reference = dataframe
        dataframe.to_csv(self.cfg["asfm"]["gcp_ref"], header=True, index=False)

    def optimize_cameras(self,
                         progress_callback: Callable = percentage_callback):
        """Function to optimize the cameras
        """
        # Disable camera locations as reference if specified in YML
        n_cameras = len(self.doc.chunk.cameras)
        for i in range(0, n_cameras):
            self.doc.chunk.cameras[i].reference.enabled = False

        self.doc.chunk.optimizeCameras(fit_f=True,
                                       fit_cx=True,
                                       fit_cy=True,
                                       fit_b1=True,
                                       fit_b2=True,
                                       fit_k1=True,
                                       fit_k2=True,
                                       fit_k3=True,
                                       fit_k4=True,
                                       fit_p1=True,
                                       fit_p2=True,
                                       fit_corrections=False,
                                       adaptive_fitting=True,
                                       tiepoint_covariance=False,
                                       progress=progress_callback)

        self.save_project()

    def reset_region(self):
        '''
        Reset the region and make it much larger than the points; necessary because if points go outside the region, they get clipped when saving
        '''

        self.doc.chunk.resetRegion()
        region_dims = self.doc.chunk.region.size
        region_dims[2] *= 3
        self.doc.chunk.region.size = region_dims

        return True

    def align_photos(self,
                     progress_callback: Callable = percentage_callback,
                     chunk: int = 0,
                     correct: bool = False):
        """Align Photos
        """
        self.save_project()
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus
        log.info("Matching photos.")

        self.doc.chunks[chunk].matchPhotos(
            downscale=self.cfg["asfm"]["align_photos"]["downscale"],
            generic_preselection=True,
            reference_preselection=True,
            reference_preselection_mode=ms.ReferencePreselectionSource,
            filter_mask=self.cfg["asfm"]["use_masking"],
            mask_tiepoints=True,
            keypoint_limit=40000,
            tiepoint_limit=4000,
            keep_keypoints=False,
            cameras=self.doc.chunks[chunk].cameras,
            guided_matching=False,
            reset_matches=False,
            subdivide_task=False,
            workitem_size_cameras=20,
            workitem_size_pairs=80,
            max_workgroup_size=100,
            progress=progress_callback)

        log.info("Aligning cameras.")
        self.doc.chunks[chunk].alignCameras(
            cameras=self.doc.chunks[chunk].cameras,
            min_image=2,
            adaptive_fitting=True,  # adaptive_fitting=False,
            reset_alignment=False,
            subdivide_task=False,
            progress=progress_callback)

        unaligned_cameras = [
            camera for camera in self.doc.chunks[chunk].cameras
            if camera.transform is None
        ]

        if len(unaligned_cameras) != 0:
            log.warning(f"Found {len(unaligned_cameras)} unaligned cameras.")

        if ms.app.gpu_mask:
            ms.app.cpu_enable = True
        if self.cfg["asfm"]["align_photos"]["autosave"]:
            self.save_project()

        if correct:
            # Check if all the cameras were aligned
            log.warning("Correction enabled. Checking for unaligned cameras.")
            if len(unaligned_cameras) < 1:
                log.info(
                    "Not enough unaligned cameras to perform alignment, skipping."
                )
            if len(unaligned_cameras) > 1:
                log.info("Attempting to align unaligned cameras.")
                # Add a chunk and try to process separately
                # self.doc.addChunk()
                photos = [camera.photo.path for camera in unaligned_cameras]
                # self.doc.chunks[1].addPhotos(photos)
                self.doc.chunk.addPhotos(photos)

                # Detect markers for the new chunk
                log.info("Detecting markers.")
                # self.detect_markers(chunk=chunk + 1)
                self.detect_markers(chunk=chunk)

                # Import reference for the new chunk
                log.info("Importing reference.")
                # self.import_reference(chunk=chunk + 1)
                self.import_reference(chunk=chunk)

                # Align again
                log.info("Aligning photos agains.")
                # self.align_photos(chunk=chunk + 1, correct=False)
                self.align_photos(chunk=chunk, correct=False)

                # Merge the chunks
                log.info("Merging Chunks.")
                # c = [self.doc.chunks[-2], self.doc.chunks[-1]]
                # keys = [x.key for x in c]
                # self.doc.mergeChunks(merge_markers=True, chunks=keys)
                # Merge the chunks
                self.doc.mergeChunks(merge_markers=True, chunks=[0, 1])

                # Set the active chunk
                log.info("Setting active chunk.")
                self.doc.chunk = self.doc.chunks[chunk]
                log.warning(len(self.doc.chunk.cameras))

                # Remove duplicate unaligned cameras
                camera_counts = Counter(
                    [camera.label for camera in self.doc.chunk.cameras])

                for camera in self.doc.chunk.cameras:
                    if camera.transform is None and camera_counts[
                            camera.label] > 1:
                        self.doc.chunk.remove(camera)

                # Check for unaligned cameras again
                # unaligned_cameras_chunk = [
                #     camera for camera in self.doc.chunks[chunk].cameras
                #     if camera.transform is None
                # ]
                # log.warning(
                #     f"Found {len(unaligned_cameras_chunk)} unaligned cameras.")

                # unaligned_cameras_chunk_plus_1 = [
                #     camera for camera in self.doc.chunks[chunk + 1].cameras
                #     if camera.transform is None
                # ]
                # log.warning(
                #     f"Found {len(unaligned_cameras_chunk_plus_1)} unaligned cameras (chunk + 1)."
                # )

                # unaligned_cameras_chunk_plus_2 = [
                #     camera for camera in self.doc.chunks[chunk + 2].cameras
                #     if camera.transform is None
                # ]
                # log.warning(
                #     f"Found {len(unaligned_cameras_chunk_plus_2)} unaligned cameras (chunk + 2)."
                # )

                self.save_project()

    def build_depth_map(self,
                        progress_callback: Callable = percentage_callback):
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus
        log.info("Number of cameras in chunk at depth map: ",
                 len(self.doc.chunk.cameras))

        self.doc.chunk.buildDepthMaps(
            downscale=self.cfg["asfm"]["depth_map"]["downscale"],
            filter_mode=ms.MildFiltering,
            cameras=self.doc.chunk.cameras,
            reuse_depth=False,
            max_neighbors=-1,
            subdivide_task=True,
            workitem_size_cameras=20,
            max_workgroup_size=100,
            progress=progress_callback)
        if ms.app.gpu_mask:
            ms.app.cpu_enable = True
        if self.cfg["asfm"]["depth_map"]["autosave"]:
            self.save_project()

    def build_dense_cloud(self,
                          progress_callback: Callable = percentage_callback):
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        if self.doc.chunk.depth_maps is None:
            self.build_depth_map()

        self.doc.chunk.buildDenseCloud(point_colors=True,
                                       point_confidence=False,
                                       keep_depth=True,
                                       max_neighbors=100,
                                       subdivide_task=True,
                                       workitem_size_cameras=20,
                                       max_workgroup_size=100,
                                       progress=progress_callback)
        if ms.app.gpu_mask:
            ms.app.cpu_enable = True
        if self.cfg["asfm"]["dense_cloud"]["autosave"]:
            self.save_project()

    def build_dem(self, progress_callback: Callable = percentage_callback):

        if self.doc.chunk.dense_cloud is None:
            self.build_dense_cloud()

        self.doc.chunk.buildDem(source_data=ms.DenseCloudData,
                                interpolation=ms.EnabledInterpolation,
                                flip_x=False,
                                flip_y=False,
                                flip_z=False,
                                resolution=0,
                                subdivide_task=True,
                                workitem_size_tiles=10,
                                max_workgroup_size=100,
                                progress=progress_callback)
        if self.cfg["asfm"]["dem"]["autosave"]:
            self.save_project()

        if self.cfg["asfm"]["dem"]["export"]["enabled"]:
            image_compression = ms.ImageCompression()
            image_compression.tiff_big = True
            kwargs = {"image_compression": image_compression}

            self.doc.chunk.exportRaster(path=self.cfg["asfm"]["dem_path"],
                                        image_format=ms.ImageFormatTIFF,
                                        source_data=ms.ElevationData,
                                        progress=progress_callback,
                                        **kwargs)

    def build_ortomosaic(self,
                         progress_callback: Callable = percentage_callback):

        self.doc.chunk.buildOrthomosaic(surface_data=ms.ElevationData,
                                        blending_mode=ms.MosaicBlending,
                                        fill_holes=True,
                                        ghosting_filter=False,
                                        cull_faces=False,
                                        refine_seamlines=False,
                                        resolution=0,
                                        resolution_x=0,
                                        resolution_y=0,
                                        flip_x=False,
                                        flip_y=False,
                                        flip_z=False,
                                        subdivide_task=True,
                                        workitem_size_cameras=20,
                                        workitem_size_tiles=10,
                                        max_workgroup_size=100,
                                        progress=progress_callback)
        if self.cfg["asfm"]["orthomosaic"]["autosave"]:
            self.save_project()

        if self.cfg["asfm"]["orthomosaic"]["export"]["enabled"]:

            image_compression = ms.ImageCompression()
            image_compression.tiff_big = True

            self.doc.chunk.exportRaster(path=self.cfg["asfm"]["ortho_path"],
                                        image_format=ms.ImageFormatTIFF,
                                        source_data=ms.OrthomosaicData,
                                        progress=progress_callback,
                                        image_compression=image_compression)

    def camera_paramters(self, camera):

        row = dict()
        row["f"] = camera.calibration.f  # Focal length in pixels
        row["cx"] = camera.calibration.cx
        row["cy"] = camera.calibration.cy
        row["k1"] = camera.calibration.k1
        row["k2"] = camera.calibration.k2
        row["k3"] = camera.calibration.k3
        row["k4"] = camera.calibration.k4
        row["p1"] = camera.calibration.p1
        row["p2"] = camera.calibration.p2
        row["b1"] = camera.calibration.b1
        row["b2"] = camera.calibration.b2
        row["pixel_height"] = camera.sensor.pixel_height
        row["pixel_width"] = camera.sensor.pixel_width

        return row

    def export_stats(self):

        # Percentage of aligned images
        total_cameras = len(self.doc.chunk.cameras)
        aligned_cameras = 0
        for row in self.camera_reference.content_dict:
            aligned_cameras += row["Alignment"]
        percentage_aligned_cameras = aligned_cameras / total_cameras

        # Percentage of detected markers
        total_gcps = len(self.gcp_reference)
        detected_gcps = 0
        for row in self.gcp_reference.content_dict:
            detected_gcps += row["Detected"]
        percentage_detected_gcps = detected_gcps / total_gcps

        dataframe = DataFrame(
            [{
                "Total_Cameras": total_cameras,
                "Aligned_Cameras": aligned_cameras,
                "Percentage_Aligned_Cameras": percentage_aligned_cameras,
                "Total_GCPs": total_gcps,
                "Detected_GCPs": detected_gcps,
                "Percentage_Detected_GCPs": percentage_detected_gcps
            }], "Total_Cameras")
        self.error_statistics = dataframe
        dataframe.to_csv(self.cfg["asfm"]["err_ref"], header=True, index=False)

    def camera_fov(self):

        rows = []
        row_template = {
            "label": "",
            "top_left_x": "",
            "top_left_y": "",
            "bottom_left_x": "",
            "bottom_left_y": "",
            "bottom_right_x": "",
            "bottom_right_y": "",
            "top_right_x": "",
            "top_right_y": "",
            "height": "",
            "width": ""
        }

        for camera in self.doc.chunk.cameras:

            row = deepcopy(row_template)
            calculate_fov = True

            row["label"] = camera.label

            pixel_height = camera.sensor.pixel_height
            if pixel_height is None:
                print(
                    f"pixel_height missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            pixel_width = camera.sensor.pixel_width
            if pixel_width is None:
                print(
                    f"pixel_width missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            height = camera.sensor.height
            if height is None:
                print(
                    f"height missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            width = camera.sensor.width
            if width is None:
                print(
                    f"width missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            f = camera.calibration.f
            if f is None:
                print(
                    f"f missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            if calculate_fov:
                # Convert focal length and image dimensions
                # to real-world units
                f_height = f * pixel_height
                f_width = f * pixel_width
                image_height = height * pixel_height
                image_width = width * pixel_width

                # Find the actual object height and width
                camera_height = self.camera_reference.retrieve(
                    camera.label, "Estimated_Z")

                if not camera_height:
                    continue
                object_half_height = find_object_dimension(
                    f_height, image_height / 2, camera_height)
                object_half_width = find_object_dimension(
                    f_width, image_width / 2, camera_height)

                row["height"] = 2. * object_half_height
                row["width"] = 2. * object_half_width

                # Find the field of view coordinates in the rotated
                yaw_angle = self.camera_reference.retrieve(
                    camera.label, "Estimated_Yaw")

                center_x = self.camera_reference.retrieve(
                    camera.label, "Estimated_X")
                center_y = self.camera_reference.retrieve(
                    camera.label, "Estimated_Y")
                if not yaw_angle or not center_x or not center_y:
                    continue

                center_coords = [center_x, center_y]

                top_left_x, top_left_y, \
                bottom_left_x, bottom_left_y, \
                bottom_right_x, bottom_right_y, \
                top_right_x, top_right_y = field_of_view(center_coords, object_half_width, object_half_height, yaw_angle)

                row["top_left_x"], row["top_left_y"] = top_left_x, top_left_y
                row["bottom_left_x"], row[
                    "bottom_left_y"] = bottom_left_x, bottom_left_y
                row["bottom_right_x"], row[
                    "bottom_right_y"] = bottom_right_x, bottom_right_y
                row["top_right_x"], row[
                    "top_right_y"] = top_right_x, top_right_y

            rows.append(row)

        df = DataFrame(rows, "label")

        df.to_csv(self.cfg["asfm"]["fov_ref"], index=False, header=True)

    def capture_view(self):
        ortho_fname = self.cfg["asfm"]["ortho_path"]
        save_path = self.cfg["asfm"]["preview"]
        # # Open the file:
        with rio.open(ortho_fname, "r") as raster:
            # Read the grid values into numpy arrays
            red = raster.read(1, masked=True)
            green = raster.read(2, masked=True)
            blue = raster.read(3, masked=True)
            alpha = raster.read(4, masked=True)
            # Create RGBA natural color composite
            rgba = np.dstack((red, green, blue, alpha))
            width = self.cfg["asfm"]["max_wid"]
            r = float(width) / rgba.shape[1]
            dim = (width, int(rgba.shape[0] * r))
            # perform the actual resizing of the image
            resized = cv2.resize(rgba, dim, interpolation=cv2.INTER_AREA)
            pilimg = Image.fromarray(resized)
            dpi = self.cfg["asfm"]["dpi"]
            pilimg.save(save_path, dpi=(dpi, dpi))
