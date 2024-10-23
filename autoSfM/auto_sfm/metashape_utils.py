import glob
from pathlib import Path
import logging
import os
from collections import Counter
from copy import deepcopy
from typing import Callable

import Metashape as ms

from .callbacks import percentage_callback
from .dataframe import DataFrame
from .estimation import (CameraStats, MarkerStats, field_of_view,
                         find_object_dimension)

log = logging.getLogger(__name__)


class SfM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_id = self.cfg.general.batch_id.split("_")[0]
        self.crs = self.get_crs_str(cfg)
        self.markerbit = self.get_target_bit(cfg)
        self.metashape_key = cfg.general.metashape_key
        self.doc = self.load_or_create_project(cfg["asfm"]["proj_path"])
        self.metashape_key = cfg.general.metashape_key
        self.num_gpus = (
            cfg["asfm"]["num_gpus"]
            if cfg["asfm"]["num_gpus"] != "all"
            else 2 ** len(ms.app.enumGPUDevices()) - 1
        )
    def _remove_duplicate_and_unaligned_cameras(self):
        """Removes duplicate aligned cameras and unaligned cameras from the chunk."""
        unique_aligned_cameras = set()
        cameras_to_remove = []

        for camera in self.doc.chunk.cameras:
            if camera.transform is None:
                cameras_to_remove.append(camera)
            elif camera.label in unique_aligned_cameras:
                cameras_to_remove.append(camera)
            else:
                unique_aligned_cameras.add(camera.label)

        for camera in cameras_to_remove:
            self.doc.chunk.remove(camera)

        log.info(f"Unaligned cameras in current chunk: {len(self.get_unaligned_cameras())}")
        log.info(f"Final number of cameras in current chunk after realignment: {len(self.doc.chunk.cameras)}")

    def get_camera_stats(self, show=True):
        """Get the number of aligned, unaligned, and duplicate cameras for each chunk."""
        stats = []
        for chunk in self.doc.chunks:
            chunk_stats = {
                "chunk_label": chunk.label,
                "aligned_cameras": 0,
                "unaligned_cameras": 0,
                "duplicate_cameras": 0,
                "tiepoints": 0,
            }
            f = ms.TiePoints.Filter()
            f.init(chunk, criterion = ms.TiePoints.Filter.ReprojectionError)
            f.values
            chunk_stats["tiepoints"] = len(f.values) #chunk.tie_points.values

            unique_aligned_cameras = set()
            for camera in chunk.cameras:
                if camera.transform is None:
                    chunk_stats["unaligned_cameras"] += 1
                elif camera.label in unique_aligned_cameras:
                    chunk_stats["duplicate_cameras"] += 1
                else:
                    unique_aligned_cameras.add(camera.label)
                    chunk_stats["aligned_cameras"] += 1
            stats.append(chunk_stats)
            if show:
                log.info(
                f"Chunk: {chunk_stats['chunk_label']}, "
                f"Aligned Cameras: {chunk_stats['aligned_cameras']}, "
                f"Unaligned Cameras: {chunk_stats['unaligned_cameras']}, "
                f"Duplicate Cameras: {chunk_stats['duplicate_cameras']}, "
                f"Chunk tiepoints: {chunk_stats['tiepoints']}"
            )
        return stats
    
    def get_target_bit(self, cfg: dict) -> ms.TargetType:
        season = cfg.general.season
        state_id = cfg.general.batch_id.split("_")[0]
        seasons_12bit = [
            "summer_weeds_2022", 
            "cool_season_covers_2022_2023", 
            "summer_cash_crops_2023", 
            "summer_weeds_2023",
            "cool_season_covers_2023_2024"
            ]
        
        markerBit = ms.CircularTarget14bit
        for season12 in seasons_12bit:
            if (season in season12) or state_id == "TX":
                markerBit = ms.CircularTarget12bit
            
        return markerBit
    
    def get_crs_str(self, cfg):
        season = cfg.general.season
        state_id = cfg.general.batch_id.split("_")[0]
        local_crs = [
            "summer_weeds_2022", 
            "cool_season_covers_2022_2023", 
            "summer_cash_crops_2023", 
            "summer_weeds_2023",
            "cool_season_covers_2023_2024"
            ]
        if state_id == "MD" and season not in local_crs:
            crs = "EPSG::4326"
        elif state_id == "NC" and season not in local_crs:
            crs = "EPSG::4326"
        else:
            crs = "LOCAL"

        log.info(f"CRS: {crs}")
        return crs
        
    
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
            log.info(f"Metashape project already exisits. Opening project file.")
            # Metashape window
            doc.open(project_path, read_only=False, ignore_lock=True)
        else:
            # Create a project and save
            log.info(f"Creating new Metashape project for {batch_id}")
            doc.addChunk()
            doc.save(project_path)

        return doc

    def save_project(self):
        """Save the project"""
        project_path = self.cfg["asfm"]["proj_path"]
        assert project_path.split(".")[-1] == "psx"
        self.doc.save()

    def add_photos(self):
        """Adds a directory to the project"""
        photos = glob.glob(os.path.join(self.cfg["asfm"]["down_photos"], "*.jpg"))
        if self.doc.chunk is None:
            self.doc.addChunk()
        self.doc.chunk.crs = ms.CoordinateSystem(self.crs)
        self.doc.chunk.addPhotos(photos)

    def add_masks(self):
        """Adds masks to the cameras"""
        self.doc.chunk.generateMasks(
            path=self.cfg["asfm"]["down_masks"] + "/{filename}_mask.png",
            masking_mode=ms.MaskingMode.MaskingModeFile,
            cameras=self.doc.chunk.cameras,
        )
        self.save_project()

    def detect_markers(
        self, chunk: int = 0, progress_callback: Callable = percentage_callback
    ):
        """Detects 12 or 14 bit circular markers"""
        log.info(f"Detecting markers: {self.markerbit}")
        self.doc.chunks[chunk].detectMarkers(
            target_type=self.markerbit,
            tolerance=50,
            filter_mask=False,
            inverted=False,
            noparity=False,
            maximum_residual=5,
            minimum_size=0,
            minimum_dist=5,
            cameras=self.doc.chunks[chunk].cameras,
            progress=progress_callback,
        )
        self.save_project()

    def import_reference(self, chunk: int = 0):
        """Imports reference points"""
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
            shutter_lag=0,
            crs=ms.CoordinateSystem(self.crs)
        )
        self.save_project()

    def export_camera_reference(self):
        """Exports the reference to a CSV file."""
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

    def optimize_cameras(self, progress_callback: Callable = percentage_callback):
        """Function to optimize the cameras"""
        # Disable camera locations as reference if specified in YML
        n_cameras = len(self.doc.chunk.cameras)
        for i in range(0, n_cameras):
            self.doc.chunk.cameras[i].reference.enabled = False

        self.doc.chunk.optimizeCameras(
            fit_f=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_f"],
            fit_cx=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_cx"],
            fit_cy=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_cy"],
            fit_b1=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_b1"],
            fit_b2=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_b2"],
            fit_k1=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_k1"],
            fit_k2=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_k2"],
            fit_k3=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_k3"],
            fit_k4=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_k4"],
            fit_p1=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_p1"],
            fit_p2=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_p2"],
            fit_corrections=self.cfg["asfm"]["optimize_cameras_cfg"]["fit_corrections"],
            adaptive_fitting=self.cfg["asfm"]["optimize_cameras_cfg"]["adaptive_fitting"],
            tiepoint_covariance=self.cfg["asfm"]["optimize_cameras_cfg"]["tiepoint_covariance"],
            progress=progress_callback,
        )

        self.save_project()

    def get_unaligned_cameras(
        self,
        chunk: int = 0,
    ):
        unaligned_cameras = [
            camera
            for camera in self.doc.chunks[chunk].cameras
            if camera.transform is None
        ]
        return unaligned_cameras

    def reset_region(self):
        """
        Reset the region and make it much larger than the points; necessary because if points go outside the region, they get clipped when saving
        """

        self.doc.chunk.resetRegion()
        region_dims = self.doc.chunk.region.size
        region_dims[2] *= 3
        self.doc.chunk.region.size = region_dims

        return True

    def match_photos(
            self,
            progress_callback: Callable = percentage_callback,
            chunk: int = 0,
            reference_preselection=ms.ReferencePreselectionSource):
        """Matches photos in the specified chunk using the provided settings."""
        log.info("Matching photos")

        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus
        match_settings = self.cfg["asfm"]["align_photos"]
        self.doc.chunks[chunk].matchPhotos(
            downscale=match_settings["downscale"],
            generic_preselection=match_settings["generic_preselection"],
            reference_preselection=match_settings["reference_preselection"],
            reference_preselection_mode=reference_preselection,
            filter_mask=self.cfg["asfm"]["use_masking"],
            mask_tiepoints=True,
            filter_stationary_points=match_settings["filter_stationary_points"],
            keypoint_limit=600000,
            keypoint_limit_per_mpx=10000,
            tiepoint_limit=200000,
            keep_keypoints=False,
            cameras=self.doc.chunks[chunk].cameras,
            guided_matching=False,
            reset_matches=True,
            subdivide_task=True,
            workitem_size_cameras=20,
            workitem_size_pairs=80,
            max_workgroup_size=100,
            progress=progress_callback,
        )

        ms.app.cpu_enable = True if ms.app.gpu_mask else False
        self.save_project()

    def align_photos(
            self,
            progress_callback: Callable = percentage_callback,
            chunk: int = 0,
            correct: bool = False,
            ):
        
        
        """Aligns photos in the specified chunk and optionally corrects unaligned cameras."""
        log.info("Aligning photos")
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        self.doc.chunks[chunk].alignCameras(
            cameras=self.doc.chunks[chunk].cameras,
            min_image=2,
            adaptive_fitting=False,
            reset_alignment=True,
            subdivide_task=True,
            progress=progress_callback,
        )
        ms.app.cpu_enable = True if ms.app.gpu_mask else False
        self.save_project()

        unaligned_cameras = self.get_unaligned_cameras(chunk)
        if unaligned_cameras:
            log.warning(f"Found {len(unaligned_cameras)} unaligned cameras.")
            if correct:
                prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                self._correct_unaligned_cameras(unaligned_cameras, chunk, progress_callback)
                unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)      
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)  
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)  
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)     
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)  

                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)  
                if prev_unaligned_cameras != unaligned_cameras:
                    prev_unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)
                    self._correct_unaligned_cameras(unaligned_cameras=unaligned_cameras, progress_callback=progress_callback, chunk=len(self.doc.chunks) - 1)
                    unaligned_cameras = self.get_unaligned_cameras(chunk=len(self.doc.chunks) - 1)            

        self._remove_duplicate_and_unaligned_cameras()
        self.reset_region()
        self.save_project()
    
    def _correct_unaligned_cameras(self, unaligned_cameras, chunk, progress_callback):
        """Attempts to correct unaligned cameras by reprocessing them."""
        log.warning("Correction enabled. Checking for unaligned cameras.")
        
        if len(unaligned_cameras) < 1:
            log.info("Not enough unaligned cameras to perform alignment, skipping.")
            return

        log.info("Attempting to align unaligned cameras.")
        new_chunk = self.doc.addChunk()
        photos = [camera.photo.path for camera in unaligned_cameras]

        new_chunk.addPhotos(photos)

        self.detect_markers(chunk=len(self.doc.chunks) - 1)
        self.import_reference(chunk=len(self.doc.chunks) - 1)

        log.info("Matching and Aligning photos again.")
        self.match_photos(chunk=len(self.doc.chunks) - 1)
        self.align_photos(chunk=len(self.doc.chunks) - 1, correct=False)

        log.info("Merging Chunks.")
        self.doc.mergeChunks(chunks=[chunk, len(self.doc.chunks) - 1], merge_markers=True, progress=progress_callback)
        log.info("Setting active chunk.")
        self.doc.chunk = self.doc.chunks[-1]

    def _remove_duplicate_and_unaligned_cameras(self):
        """Removes duplicate aligned cameras and unaligned cameras from the chunk."""
        unique_aligned_cameras = set()
        cameras_to_remove = []

        for camera in self.doc.chunk.cameras:
            if camera.transform is None:
                cameras_to_remove.append(camera)
            elif camera.label in unique_aligned_cameras:
                cameras_to_remove.append(camera)
            else:
                unique_aligned_cameras.add(camera.label)

        for camera in cameras_to_remove:
            self.doc.chunk.remove(camera)

        log.info(f"Unaligned cameras in current chunk: {len(self.get_unaligned_cameras())}")
        log.info(f"Final number of cameras in current chunk after realignment: {len(self.doc.chunk.cameras)}")


    def build_depth_map(self, progress_callback: Callable = percentage_callback):
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus
        log.info(
            f"Number of cameras in chunk at depth map: {len(self.doc.chunk.cameras)}"
        )
        log.info(f"Chunks names: {[chunk.label for chunk in self.doc.chunks]}")
        
        self.doc.chunk.buildDepthMaps(
            downscale=self.cfg["asfm"]["depth_map"]["downscale"],
            filter_mode=ms.ModerateFiltering,
            cameras=self.doc.chunk.cameras,
            reuse_depth=True,
            max_neighbors=-1,
            subdivide_task=True,
            workitem_size_cameras=20,
            max_workgroup_size=100,
            progress=progress_callback,
        )
        if ms.app.gpu_mask:
            ms.app.cpu_enable = True
        if self.cfg["asfm"]["depth_map"]["autosave"]:
            self.save_project()

    def build_dense_cloud(self, progress_callback: Callable = percentage_callback):
        ms.app.cpu_enable = False
        ms.app.gpu_mask = self.num_gpus

        if self.doc.chunk.depth_maps is None:
            self.build_depth_map()

        self.doc.chunk.buildPointCloud(
            point_colors=True,
            point_confidence=False,
            keep_depth=True,
            max_neighbors=100,
            uniform_sampling=True,
            subdivide_task=True,
            workitem_size_cameras=20,
            max_workgroup_size=100,
            progress=progress_callback,
        )
        if ms.app.gpu_mask:
            ms.app.cpu_enable = True
        if self.cfg["asfm"]["dense_cloud"]["autosave"]:
            self.save_project()

    def build_model(self, progress_callback: Callable = percentage_callback):
        self.doc.chunk.buildModel(
            surface_type=ms.Arbitrary,
            interpolation=ms.Extrapolated,
            face_count=ms.LowFaceCount,
            source_data=ms.PointCloudData,
            vertex_colors=True,
            vertex_confidence=True,
            volumetric_masks=False,
            keep_depth=True,
            trimming_radius=0,
            subdivide_task=False,
            workitem_size_cameras=20,
            max_workgroup_size=100,
            progress=progress_callback,
        )
        self.doc.chunk.model.closeHoles(level=100)
        self.save_project()

    def build_texture(self, progress_callback: Callable = percentage_callback):
        self.doc.chunk.buildUV(
            mapping_mode=ms.GenericMapping, progress=progress_callback
        )

        self.save_project()

        self.doc.chunk.buildTexture(
            texture_size=4096, ghosting_filter=True, progress=progress_callback
        )
        self.save_project()

    def build_dem(self, progress_callback: Callable = percentage_callback):
        if self.doc.chunk.point_cloud is None:
            self.build_dense_cloud()

        self.doc.chunk.buildDem(
            source_data=ms.PointCloudData,
            interpolation=ms.EnabledInterpolation,
            flip_x=False,
            flip_y=False,
            flip_z=False,
            resolution=0,
            subdivide_task=True,
            workitem_size_tiles=10,
            max_workgroup_size=100,
            progress=progress_callback,
        )
        if self.cfg["asfm"]["dem"]["autosave"]:
            self.save_project()

        if self.cfg["asfm"]["dem"]["export"]["enabled"]:
            image_compression = ms.ImageCompression()
            image_compression.tiff_big = True
            kwargs = {"image_compression": image_compression}

            self.doc.chunk.exportRaster(
                path=self.cfg["asfm"]["dem_path"],
                image_format=ms.ImageFormatTIFF,
                source_data=ms.ElevationData,
                progress=progress_callback,
                **kwargs,
            )

    def build_ortomosaic(self, progress_callback: Callable = percentage_callback):
        self.doc.chunk.buildOrthomosaic(
            surface_data=ms.ElevationData,
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
            progress=progress_callback,
        )
        if self.cfg["asfm"]["orthomosaic"]["autosave"]:
            self.save_project()

        if self.cfg["asfm"]["orthomosaic"]["export"]["enabled"]:
            image_compression = ms.ImageCompression()
            image_compression.tiff_big = True

            self.doc.chunk.exportRaster(
                path=self.cfg["asfm"]["ortho_path"],
                image_format=ms.ImageFormatTIFF,
                source_data=ms.OrthomosaicData,
                progress=progress_callback,
                image_compression=image_compression,
            )

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
        percentage_detected_gcps = detected_gcps / max(1, total_gcps)

        dataframe = DataFrame(
            [
                {
                    "Total_Cameras": total_cameras,
                    "Aligned_Cameras": aligned_cameras,
                    "Percentage_Aligned_Cameras": percentage_aligned_cameras,
                    "Total_GCPs": total_gcps,
                    "Detected_GCPs": detected_gcps,
                    "Percentage_Detected_GCPs": percentage_detected_gcps,
                }
            ],
            "Total_Cameras",
        )
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
            "width": "",
        }

        for camera in self.doc.chunk.cameras:
            row = deepcopy(row_template)
            calculate_fov = True

            row["label"] = camera.label

            pixel_height = camera.sensor.pixel_height
            if pixel_height is None:
                log.warning(
                    f"pixel_height missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            pixel_width = camera.sensor.pixel_width
            if pixel_width is None:
                log.warning(
                    f"pixel_width missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            height = camera.sensor.height
            if height is None:
                log.warning(
                    f"height missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            width = camera.sensor.width
            if width is None:
                log.warning(
                    f"width missing for camera {camera.label}, skipping FOV calculation."
                )
                calculate_fov = False

            f = camera.calibration.f
            if f is None:
                log.warning(f"f missing for camera {camera.label}, skipping FOV calculation.")
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
                    camera.label, "Estimated_Z"
                )

                if not camera_height:
                    continue
                object_half_height = find_object_dimension(
                    f_height, image_height / 2, camera_height
                )
                object_half_width = find_object_dimension(
                    f_width, image_width / 2, camera_height
                )

                row["height"] = 2.0 * object_half_height
                row["width"] = 2.0 * object_half_width

                # Find the field of view coordinates in the rotated
                yaw_angle = self.camera_reference.retrieve(
                    camera.label, "Estimated_Yaw"
                )

                center_x = self.camera_reference.retrieve(camera.label, "Estimated_X")
                center_y = self.camera_reference.retrieve(camera.label, "Estimated_Y")
                if not yaw_angle or not center_x or not center_y:
                    continue

                center_coords = [center_x, center_y]

                (
                    top_left_x,
                    top_left_y,
                    bottom_left_x,
                    bottom_left_y,
                    bottom_right_x,
                    bottom_right_y,
                    top_right_x,
                    top_right_y,
                ) = field_of_view(
                    center_coords, object_half_width, object_half_height, yaw_angle
                )

                row["top_left_x"], row["top_left_y"] = top_left_x, top_left_y
                row["bottom_left_x"], row["bottom_left_y"] = (
                    bottom_left_x,
                    bottom_left_y,
                )
                row["bottom_right_x"], row["bottom_right_y"] = (
                    bottom_right_x,
                    bottom_right_y,
                )
                row["top_right_x"], row["top_right_y"] = top_right_x, top_right_y

            rows.append(row)

        df = DataFrame(rows, "label")

        df.to_csv(self.cfg["asfm"]["fov_ref"], index=False, header=True)

    def export_report(self, progress_callback: Callable = percentage_callback):
        self.doc.chunk.exportReport(
            path=f"{self.cfg['batchdata']['autosfm']}/{self.cfg['general']['batch_id']}.pdf",
            title=self.cfg["general"]["batch_id"],
            description="report",
            font_size=12,
            page_numbers=True,
            include_system_info=True,
            progress=progress_callback,
        )
