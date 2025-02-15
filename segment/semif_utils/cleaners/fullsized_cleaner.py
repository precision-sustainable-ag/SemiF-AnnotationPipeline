import logging
from pathlib import Path
from pprint import pprint
from omegaconf import DictConfig
import json
from typing import Dict, Any, List
import numpy as np
from semif_utils.utils import (
    match_season_to_date_ranges, 
    match_single_date_range_to_season, 
    find_date_range_season
    )

log = logging.getLogger(__name__)

def safe_get(d: dict, key: str, default=None):
    """Safely get a value from a dictionary with thorough logging and error handling."""
    try:
        value = d.get(key, default)
        if value is None:
            log.warning(f"Key '{key}' not found in dictionary. Returning default value.")
        else:
            log.debug(f"Key '{key}' found in dictionary with value: {value}")
        return value
    except Exception as e:
        log.error(f"Error accessing key '{key}' in dictionary: {e}")
        return default
    
def read_json(json_path: Path, suppress_error=False) -> dict:
    try:
        with open(json_path, "r") as json_file:
            return json.load(json_file)
    except FileNotFoundError as e:
        log.error(f"Could not read metadata file: {json_path} .")

    except json.JSONDecodeError as e:
        if not suppress_error:
            log.error(f"Could not decode JSON file: {json_path} .")
        return {}


class AnnotationCleaner:
    """
    Cleans annotations for the v3 schema.
    """
    def __init__(self,cfg: DictConfig, metadata_path: Path):
        self.root_data_dir = Path("data", "working")
        self.cutout_data_type = "semifield-cutouts"
        self.species_info = read_json(cfg.data.species)

        self.missing_cutout_ids = set()
        self.missing_bbox_xywh = set()
        self.missing_category_class_ids = set()
        self.missing_is_primary = set()
        self.missing_overlapping_cutout_ids = set()
        self.metadata_path = Path(metadata_path)
        self.pipeline = [
            self._get_cutout_id,
            self._get_bbox_xywh,
            self._get_category_class_id,
            self._get_is_primary,
            self._get_overlapping_cutout_ids,
            self._get_cutout_exists,
            self._get_non_target_weed,
        ]
        self.class_ids = []

    def _get_cutout_paths(self) -> None:
        """Get the cutout paths from the metadata."""
        image_id = self.metadata_path.stem
        batch_id = self.metadata_path.parent.parent.name
        cutout_batch_dir = Path(self.root_data_dir, self.cutout_data_type, batch_id)
        image_cutouts = cutout_batch_dir.glob(f"{image_id}*.json")
        return list(image_cutouts)
    
    def clean_annotations(self, annotation: Dict[str, Any]) -> dict:
        """Apply only the cleaning methods defined in the pipeline."""
        self.annotation = annotation
        
        for clean_method in self.pipeline:
            if not callable(clean_method):
                method = getattr(self, f"_{clean_method}", None)
            else:
                method = clean_method
            
            if method:
                try:
                    method()
                except Exception as e:
                    log.exception(f"Error in {clean_method}: {e}")
            else:
                log.error(f"Method {clean_method} not found in annotation pipeline. Check config for spelling errors.")
    
        return self.annotation

    def _get_bbox_xywh(self) -> None:
        """Get the bbox_xywh from the annotation."""
        found_bbox_xywh_flag = False
        bbox_xywh = self.annotation.get("bbox_xywh")
        if bbox_xywh:
            found_bbox_xywh_flag = True
            log.debug(f"bbox_xywh key found in annotation: {bbox_xywh}")

        elif not bbox_xywh:
            local_coordinates = self.annotation.get("local_coordinates")
            if local_coordinates:
                tl_xy = local_coordinates["top_left"]
                tr_xy = local_coordinates["top_right"]
                br_xy = local_coordinates["bottom_right"]
                image_width = 9560
                image_height = 6368
                
                if 0 <= tl_xy[0] <=1 and 0 <= tl_xy[1] <=1 and 0 <= tr_xy[0] <=1 and 0 <= tr_xy[1] <=1:
                    bbox_x = int(tl_xy[0] * image_width)
                    bbox_y = int(tl_xy[1] * image_height)
                
                    bbox_width = int((tr_xy[0] - tl_xy[0]) * image_width)
                    bbox_height = int((br_xy[1] - tr_xy[1]) * image_height)
                else:
                    bbox_x = int(tl_xy[0]) #//image_width
                    bbox_y = int(tl_xy[1]) #//image_height
                
                    bbox_width = int((tr_xy[0] - tl_xy[0])) #// image_width
                    bbox_height =int((br_xy[1] - tr_xy[1])) #// image_height

                
                bbox_xywh = [bbox_x, bbox_y, bbox_width, bbox_height] # (top left x, top left y, bbox_widht, bbox_height)
                found_bbox_xywh_flag = True

        if found_bbox_xywh_flag:
            self.annotation['bbox_xywh'] = bbox_xywh
            log.debug(f"bbox_xywh key found in annotation: {bbox_xywh}")
        
        else:
            if self.metadata_path not in self.missing_bbox_xywh:
                self.missing_bbox_xywh.add(self.metadata_path)
                log.error(f"bbox_xywh key not found in annotation. {self.metadata_path} .")

    def _get_cutout_id(self) -> None:
        
        cutout_id = self.annotation.get("cutout_id")
        if cutout_id:
            log.debug(f"cutout_id key found in annotation: {cutout_id}")
        
        elif not cutout_id:    
            cutout_id = self.annotation.get("bbox_id")
            if cutout_id:
                log.debug(f"bbox_id key found in annotation: {cutout_id}")

        if cutout_id:
            self.annotation['cutout_id'] = cutout_id
            log.debug(f"cutout_id key found in metadata: {cutout_id} ")
        
        else:
            if self.metadata_path not in self.missing_cutout_ids:
                log.error(f"cutout_id key not found in metadata. {self.metadata_path} .")
                self.missing_cutout_ids.add(self.metadata_path)
    
    def _get_category_class_id(self) -> None:
        """Get the category_class_id from the annotation."""
        category_class_id = self.annotation.get("category_class_id")
        if not category_class_id:
            category_class = self.annotation.get("cls") #, self.annotation.get("category"))
            if category_class == "plant":
                category = self.species_info['species'][category_class]
                category_class_id = category.get("class_id")
            
            elif isinstance(category_class, dict):
                category_class_id = category_class.get("class_id")
                
        if category_class_id:
            self.annotation['category_class_id'] = category_class_id
            self.class_ids.append(category_class_id)
        
        else:
            if self.metadata_path not in self.missing_category_class_ids:
                self.missing_category_class_ids.add(self.metadata_path)
                log.error(f"category_class_id key not found in annotation. {self.metadata_path} .")

    def _get_is_primary(self) -> None:
        """Get the is_primary from the annotation."""
        is_primary = self.annotation.get("is_primary")
        if is_primary is None:
            cutout_id = self.annotation.get("cutout_id")
            if cutout_id:
                image_cutouts = self._get_cutout_paths()
                for cutout_path in image_cutouts:
                    cutout_stem = cutout_path.stem
                    if cutout_stem == cutout_id:
                        cutout = read_json(cutout_path)
                        is_primary = cutout.get("is_primary")
        
        if is_primary is not None:
            self.annotation["is_primary"] = is_primary
        else:
            if self.metadata_path not in self.missing_is_primary:
                self.missing_is_primary.add(self.metadata_path)
                log.error(f"is_primary key not found in annotation. {self.metadata_path} .")

    def _get_overlapping_cutout_ids(self) -> None:
        """Get the overlapping cutout ids from the annotation."""
        overlapping_cutout_ids = self.annotation.get("overlapping_cutout_ids")
        
        if overlapping_cutout_ids is None:
            overlapping_cutout_ids = self.annotation.get("overlapping_bbox_ids")
        
        if overlapping_cutout_ids or overlapping_cutout_ids == []:
            self.annotation["overlapping_cutout_ids"] = overlapping_cutout_ids
        else:
            
            if self.metadata_path not in self.missing_overlapping_cutout_ids:
                self.missing_overlapping_cutout_ids.add(self.metadata_path)
                log.error(f"overlapping_cutout_ids key not found in annotation. {self.metadata_path} .")

    def _get_cutout_exists(self) -> None:
        """Get the cutout_exists from the annotation."""
        batch_id = self.metadata_path.parent.parent.name
        cutout_batch_dir = Path(self.root_data_dir, self.cutout_data_type, batch_id)
        cutout_id = self.annotation.get("cutout_id")
        if cutout_id:
            cutout_path = cutout_batch_dir / f"{cutout_id}.json"
            cutout_exists = cutout_path.exists()
            self.annotation["cutout_exists"] = cutout_exists
        else:
            if self.metadata_path not in self.missing_cutout_ids:
                self.missing_cutout_ids.add(self.metadata_path)
                log.error(f"cutout_exists key not found in annotation. {self.metadata_path} .")
        
    def _get_non_target_weed(self) -> None:
        """Get the non_target_weed from the annotation."""
        non_target_weed = self.annotation.get("non_target_weed")
        non_target_weed_pred_conf = self.annotation.get("non_target_weed_pred_conf")
        self.annotation["non_target_weed"] = non_target_weed
        self.annotation["non_target_weed_pred_conf"] = non_target_weed_pred_conf
        
class FullsizedMetadataCleaner:
    """Class to clean and reformat metadata files systematically."""
    
    def __init__(self, cfg: DictConfig, batch_path: Path):
        """Initialize the cleaner with the metadata to be cleaned."""
        self.cfg = cfg
        self.batch_path = batch_path

        self.species_info = read_json(cfg.data.species)
        self.full_schema = read_json(Path(cfg.data.utilsdir,"species_information", "fullsized_schema.json"))
        self.date_ranges = cfg.date_ranges

        self.admin_classes = ["colorchecker", "unknown", "background"]
        
        self.annotation_key_flag = None
        self.categories_key_flag = None

        self.class_ids = None
        
        self.pipeline = [
            # Root properties
            self._get_batch_id,
            self._get_season,
            self._get_datetime,
            self._add_bbot_version_number,
            self._get_image_id,
            self._is_validated,
            self._data_version,
            self._get_exif_meta,
            self._get_camera_info,
            self._get_annotations,
            self._clean_annotations,
            self._get_categories,
            
            # Exif metadata
            self._get_exif_height_width,
            self._get_camera_make,
            self._get_camera_model,
            self._get_software_used,
            self._get_exposure_time,
            self._get_F_number,
            self._get_ISO_speed,
            self._get_exif_version,
            self._get_light_source,
            self._get_flash,
            self._get_focal_length,
            self._get_exposure_mode,
            self._get_white_balance,
            self._get_contrast,
            self._get_saturation,
            self._get_sharpness,
            self._get_lens_model,
            self._get_lens_specification,
            self._get_bodyserialnumber,
            # Camera Info
            self._get_aligned_status,
            self._get_estimated_xyz,
            self._get_estimated_pitch,
            self._get_estimated_yaw,
            self._get_estimated_roll,
            self._get_camera_coefficients,
            self._get_pixel_width_height,
            self._get_fov,
            #Organize and clean
            self._clean_categories,
            self._organize_exif_meta,
            self._organize_camera_info,
            self._organize_root_keys, 
            self._replace_nan_with_null



            

        ]

        self.possible_seasons = [
            'summer_cash_crops_2023',
            'summer_cash_crops_2024',
            'summer_cash_crops_2025',
            'summer_cash_crops_2026',
            
            'summer_weeds_2022',
            'summer_weeds_2023',
            'summer_weeds_2023_TXpos2',
            'summer_weeds_2024',
            'summer_weeds_2025',
            'summer_weeds_2026',
            
            'cool_season_covers_2022_2023',
            'cool_season_cover_2022_2023_MD_pos_2',
            'cool_season_cover_2022_2023_MD_pos_3',
            'cool_season_covers_2023_2024',
            'cool_season_covers_2024_2025',
            'cool_season_covers_2025_2026',
            'cool_season_covers_2026_2027',

        ]
    
    def clean(self, metadata_path: Path, metadata: dict, image_data: dict) -> dict:
        """Apply only the cleaning methods defined in the pipeline."""
        self.metadata = metadata
        self.metadata_path = metadata_path
        self.fov = image_data.get("fov", None)
        
        self.cam_ref = image_data.get("camera_reference", None)
                
        for clean_method in self.pipeline:
            if not callable(clean_method):
                method = getattr(self, f"_{clean_method}", None)
            else:
                method = clean_method
            
            if method:
                try:
                    method()
                except Exception as e:
                    log.exception(f"Error in {clean_method}: {e}")
            else:
                log.error(f"Method {clean_method} not found. Check config for spelling errors.")
        
        return self.metadata

#######################################################################################
################################### ROOT PROPS ########################################
#######################################################################################

    def _get_batch_id(self) -> None:
        batch_id = safe_get(self.metadata, "batch_id")
        if batch_id:
            self.metadata['batch_id'] = batch_id
            log.debug(f"batch_id key found in metadata: {batch_id}")
        else:
            log.error(f"batch_id key not found in metadata. {self.metadata_path} .")
    
    def _get_season(self) -> None:
        season = self.metadata.get('season')
        
        if not season:
            batch_id = safe_get(self.metadata, 'batch_id', default=None)
            if batch_id:
                state_id = batch_id.split("_")[0]
                batch_date = batch_id.split("_")[1]
                date_range_season = find_date_range_season(batch_date, state_id, self.date_ranges)            
                season = match_single_date_range_to_season(date_range_season, self.possible_seasons)

        if season:
            self.metadata['season'] = season
            log.debug(f"Season key found in metadata: {season}")
        
        else:
            log.error(f"Season key not found in metadata. {self.metadata_path} .")

    def _get_datetime(self) -> None:
        """Get the datetime from the metadata."""
        # datetime = self.metadata.get("datetime")
        datetime = safe_get(self.metadata["exif_meta"], "DateTime", default=None)
        if not datetime:
            log.error(f"Could not find datetime in metadata: {self.metadata_path} .")
        else:
            self.metadata["datetime"] = datetime

    def _add_bbot_version_number(self) -> None:
        """Add a version number to the metadata."""
        state_id = self.metadata['batch_id'].split("_")[0]
        season = self.metadata['season']
        # date_range = self.date_ranges.get(state_id)
        date_range = safe_get(self.date_ranges, state_id, default=None)
        date_range_season = match_season_to_date_ranges(season, date_range.keys())
        bbot_version = self.date_ranges[state_id][date_range_season]['bbot_version']
        
        if bbot_version:
            self.metadata['bbot_version'] = bbot_version
            log.debug(f"bbot_version key added to metadata: {bbot_version}")
        
        else:
            log.error(f"bbot_version key not found in metadata. {self.metadata_path} .")

    def _get_image_id(self) -> None:
        """Get the image_id from the metadata."""
        image_id = safe_get(self.metadata, "image_id")
        if image_id:
            self.metadata["image_id"] = image_id
            log.debug(f"image_id key found in metadata: {image_id}")
        else:
            log.error(f"image_id key not found in metadata. {self.metadata_path} .")

    def _is_validated(self) -> None:
        """Check if the metadata is validated."""
        self.metadata["validated"] = False
    
    def _data_version(self) -> None:
        """Check if the metadata is validated."""
        version = self.cfg.convert.versions.fullsized
        self.metadata["version"] = version
    
        
    #######################################################################################
    ################################## EXIF METADATA ######################################
    #######################################################################################

    def _get_exif_meta(self) -> None:
        """Get the exif metadata from the metadata."""
        exif_meta = safe_get(self.metadata, "exif_meta")
        if exif_meta:
            self.metadata["exif_meta"] = exif_meta
            log.debug(f"exif_meta key found in metadata.")
        else:
            log.error(f"exif_meta key not found in metadata. {self.metadata_path} .")

    def _get_exif_height_width(self) -> None:
        """Get the height and width from the exif metadata."""
        height = safe_get(self.metadata["exif_meta"], "ImageLength", default=None)
        width = safe_get(self.metadata["exif_meta"], "ImageWidth", default=None)
        if height and width:
            self.metadata["exif_meta"]["ImageLength"] = height
            self.metadata["exif_meta"]["ImageWidth"] = width
            log.debug(f"Height and width keys found in metadata: {height} x {width}")
        else:
            log.error(f"Height and width keys not found in metadata. {self.metadata_path} .")

    def _get_camera_make(self) -> None:
        """Get the camera make from the exif metadata."""
        camera_make = safe_get(self.metadata["exif_meta"], "Make", default=None)
        if camera_make:
            self.metadata["exif_meta"]["Make"] = camera_make
            log.debug(f"Camera make key found in metadata: {camera_make}")
        else:
            log.error(f"Camera make key not found in metadata. {self.metadata_path} .")

    def _get_camera_model(self) -> None:
        """Get the camera model from the exif metadata."""
        camera_model = safe_get(self.metadata["exif_meta"], "Model", default=None)
        if camera_model:
            self.metadata["exif_meta"]["Model"] = camera_model
            log.debug(f"Camera model key found in metadata: {camera_model}")
        else:
            log.error(f"Camera model key not found in metadata. {self.metadata_path} .")

    def _get_software_used(self) -> None:
        """Get the software used from the exif metadata."""
        software = safe_get(self.metadata["exif_meta"], "Software", default=None)
        if software:
            self.metadata["exif_meta"]["Software"] = software
            log.debug(f"Software key found in metadata: {software}")
        else:
            log.error(f"Software key not found in metadata. {self.metadata_path} .")

    def _get_exposure_time(self) -> None:
        """Get the exposure time from the exif metadata."""
        exposure_time = safe_get(self.metadata["exif_meta"], "ExposureTime", default=None)
        if exposure_time:
            self.metadata["exif_meta"]["ExposureTime"] = exposure_time
            log.debug(f"Exposure time key found in metadata: {exposure_time}")
        else:
            log.error(f"Exposure time key not found in metadata. {self.metadata_path} .")

    
    def _get_F_number(self) -> None:
        """Get the F number from the exif metadata."""
        F_number = safe_get(self.metadata["exif_meta"], "FNumber", default=None)
        if F_number:
            self.metadata["exif_meta"]["FNumber"] = F_number
            log.debug(f"F number key found in metadata: {F_number}")
        else:
            log.error(f"F number key not found in metadata. {self.metadata_path} .")

    def _get_ISO_speed(self) -> None:
        """Get the ISO speed from the exif metadata."""
        ISO_speed = safe_get(self.metadata["exif_meta"], "ISOSpeedRatings", default=None)
        if ISO_speed:
            self.metadata["exif_meta"]["ISOSpeedRatings"] = ISO_speed
            log.debug(f"ISO speed key found in metadata: {ISO_speed}")
        else:
            log.error(f"ISO speed key not found in metadata. {self.metadata_path} .")

    def _get_exif_version(self) -> None:
        """Get the exif version from the exif metadata."""
        exif_version = safe_get(self.metadata["exif_meta"], "ExifVersion", default=None)
        if exif_version:
            self.metadata["exif_meta"]["ExifVersion"] = exif_version
            log.debug(f"Exif version key found in metadata: {exif_version}")
        else:
            log.error(f"Exif version key not found in metadata. {self.metadata_path} .")

    def _get_light_source(self) -> None:
        """Get the light source from the exif metadata."""
        light_source = safe_get(self.metadata["exif_meta"], "LightSource", default=None)
        if light_source or light_source == 0:
            self.metadata["exif_meta"]["LightSource"] = light_source
            log.debug(f"Light source key found in metadata: {light_source}")
        else:
            log.error(f"Light source key not found in metadata. {self.metadata_path} .")

    def _get_flash(self) -> None:
        """Get the flash from the exif metadata."""
        flash = safe_get(self.metadata["exif_meta"], "Flash", default=None)
        if flash:
            self.metadata["exif_meta"]["Flash"] = flash
            log.debug(f"Flash key found in metadata: {flash}")
        else:
            log.error(f"Flash key not found in metadata. {self.metadata_path} .")

    def _get_focal_length(self) -> None:
        """Get the focal length from the exif metadata."""
        focal_length = safe_get(self.metadata["exif_meta"], "FocalLength", default=None)
        if focal_length:
            self.metadata["exif_meta"]["FocalLength"] = focal_length
            log.debug(f"Focal length key found in metadata: {focal_length}")
        else:
            log.error(f"Focal length key not found in metadata. {self.metadata_path} .")

    def _get_exposure_mode(self) -> None:
        """Get the exposure mode from the exif metadata."""
        exposure_mode = safe_get(self.metadata["exif_meta"], "ExposureMode", default=None)
        if exposure_mode:
            self.metadata["exif_meta"]["ExposureMode"] = exposure_mode
            log.debug(f"Exposure mode key found in metadata: {exposure_mode}")
        else:
            log.error(f"Exposure mode key not found in metadata. {self.metadata_path} .")
    
    def _get_white_balance(self) -> None:
        """Get the white balance from the exif metadata."""
        white_balance = safe_get(self.metadata["exif_meta"], "WhiteBalance", default=None)
        if white_balance or white_balance == 0:
            self.metadata["exif_meta"]["WhiteBalance"] = white_balance
            log.debug(f"White balance key found in metadata: {white_balance}")
        else:
            log.error(f"White balance key not found in metadata. {self.metadata_path} .")

    def _get_contrast(self) -> None:
        """Get the contrast from the exif metadata."""
        contrast = safe_get(self.metadata["exif_meta"], "Contrast", default=None)
        if contrast or contrast == 0:
            self.metadata["exif_meta"]["Contrast"] = contrast
            log.debug(f"Contrast key found in metadata: {contrast}")
        else:
            log.error(f"Contrast key not found in metadata. {self.metadata_path} .")

    def _get_saturation(self) -> None:
        """Get the saturation from the exif metadata."""
        saturation = safe_get(self.metadata["exif_meta"], "Saturation", default=None)
        if saturation or saturation == 0:
            self.metadata["exif_meta"]["Saturation"] = saturation
            log.debug(f"Saturation key found in metadata: {saturation}")
        else:
            log.error(f"Saturation key not found in metadata. {self.metadata_path} .")

    def _get_sharpness(self) -> None:
        """Get the sharpness from the exif metadata."""
        sharpness = safe_get(self.metadata["exif_meta"], "Sharpness", default=None)
        if sharpness or sharpness == 0:
            self.metadata["exif_meta"]["Sharpness"] = sharpness
            log.debug(f"Sharpness key found in metadata: {sharpness}")
        else:
            log.error(f"Sharpness key not found in metadata. {self.metadata_path} .")

    def _get_lens_model(self) -> None:
        """Get the lens model from the exif metadata."""
        lens_model = safe_get(self.metadata["exif_meta"], "LensModel", default=None)
        if lens_model:
            self.metadata["exif_meta"]["LensModel"] = lens_model
            log.debug(f"Lens model key found in metadata: {lens_model}")
        else:
            log.error(f"Lens model key not found in metadata. {self.metadata_path} .")

    def _get_lens_specification(self) -> None:
        """Get the lens specification from the exif metadata."""
        lens_specification = safe_get(self.metadata["exif_meta"], "LensSpecification", default=None)
        if lens_specification:
            self.metadata["exif_meta"]["LensSpecification"] = lens_specification
            log.debug(f"Lens specification key found in metadata: {lens_specification}")
        else:
            log.error(f"Lens specification key not found in metadata. {self.metadata_path} .")

    def _get_bodyserialnumber(self) -> None:
        """Get the body serial number from the exif metadata."""
        body_serial_number = self.metadata["exif_meta"].get("BodySerialNumber")
        
        if body_serial_number or body_serial_number is None:
            if body_serial_number is None:
                log.debug(f"Body serial number key is None for metadata. {self.metadata_path} .")
            
            self.metadata["exif_meta"]["BodySerialNumber"] = body_serial_number
            log.debug(f"Body serial number key found in metadata: {body_serial_number}")
        
    
    ################################################################################################
    ######################################## Camera Info ###########################################
    ################################################################################################


    def _get_camera_info(self) -> None:
        """Get the camera_info from the metadata."""
        camera_info = safe_get(self.metadata, "camera_info")
        if camera_info:
            self.metadata["camera_info"] = camera_info
            log.debug(f"camera_info key found in metadata.")
        else:
            log.error(f"camera_info key not found in metadata. {self.metadata_path} .")

    
    def _get_aligned_status(self) -> None:
        """Get the aligned status from the camera_info."""
        aligned_status = safe_get(self.cam_ref, "Alignment", default=None)
        if aligned_status:
            self.metadata["camera_info"]["aligned"] = aligned_status
            log.debug(f"aligned_status key found in metadata: {aligned_status}")
        else:
            log.error(f"aligned_status key not found in metadata. {self.metadata_path} .")

    def _get_estimated_xyz(self) -> None:
        """Get the estimated xyz from the camera_info."""
        estimated_x = safe_get(self.cam_ref, "Estimated_X", default=None)
        estimated_y = safe_get(self.cam_ref, "Estimated_Y", default=None)
        estimated_z = safe_get(self.cam_ref, "Estimated_Z", default=None)
        
        if estimated_x and estimated_y and estimated_z:
            self.metadata["camera_info"]["estimated_xyz"] = [estimated_x, estimated_y, estimated_z]
            log.debug(f"estimated_xyz key found in metadata: {estimated_x}, {estimated_y}, {estimated_z}")

        else:
            log.error(f"estimated_xyz key not found in metadata. {self.metadata_path} .")
    
    def _get_estimated_pitch(self) -> None:
        """Get the estimated pitch from the camera_info."""
        estimated_pitch = safe_get(self.cam_ref, "Estimated_Pitch", default=None)
        if estimated_pitch:
            log.debug(f"estimated_pitch key found in metadata: {estimated_pitch}")
        else:        
            log.error(f"estimated_pitch key not found in metadata. {self.metadata_path} .")
        
        self.metadata["camera_info"]["estimated_pitch"] = estimated_pitch
    
    def _get_estimated_yaw(self) -> None:
        """Get the estimated yaw from the camera_info."""
        estimated_yaw = safe_get(self.cam_ref, "Estimated_Yaw", default=None)
        if estimated_yaw:
            log.debug(f"estimated_yaw key found in metadata: {estimated_yaw}")
        else:
            log.error(f"estimated_yaw key not found in metadata. {self.metadata_path} .")
        self.metadata["camera_info"]["estimated_yaw"] = estimated_yaw

    def _get_estimated_roll(self) -> None:
        """Get the estimated roll from the camera_info."""
        estimated_roll = safe_get(self.cam_ref, "Estimated_Roll", default=None)
        if estimated_roll:
            log.debug(f"estimated_roll key found in metadata: {estimated_roll}")
        else:
            log.error(f"estimated_roll key not found in metadata. {self.metadata_path} .")
        self.metadata["camera_info"]["estimated_roll"] = estimated_roll
        
    def _get_camera_coefficients(self) -> None:
        """Get the camera coefficients from the camera_info."""
        
        f = safe_get(self.cam_ref, "f", default=None)
        if not f:
            log.warning(f"Camera focal length key is None for metadata. {self.metadata_path} .")
        
        cx = safe_get(self.cam_ref, "cx", default=None)
        if not cx:
            log.warning(f"Camera cx key is None for metadata. {self.metadata_path} .")

        cy = safe_get(self.cam_ref, "cy", default=None)
        if not cy:
            log.warning(f"Camera cy key is None for metadata. {self.metadata_path} .")
        
        b1 = safe_get(self.cam_ref, "b1", default=None)
        if not b1:
            log.warning(f"Camera b1 key is None for metadata. {self.metadata_path} .")
        
        b2 = safe_get(self.cam_ref, "b2", default=None)
        if not b2:
            log.warning(f"Camera b2 key is None for metadata. {self.metadata_path} .")
        
        k1 = safe_get(self.cam_ref, "k1", default=None)
        if not k1:
            log.warning(f"Camera k1 key is None for metadata. {self.metadata_path} .")
        
        k2 = safe_get(self.cam_ref, "k2", default=None)
        if not k2:
            log.warning(f"Camera k2 key is None for metadata. {self.metadata_path} .")
        
        k3 = safe_get(self.cam_ref, "k3", default=None)
        if not k3:
            log.warning(f"Camera k3 key is None for metadata. {self.metadata_path} .")
        
        k4 = safe_get(self.cam_ref, "k4", default=None)
        if not k4:
            log.warning(f"Camera k4 key is None for metadata. {self.metadata_path} .")

        p1 = safe_get(self.cam_ref, "p1", default=None)
        if not p1:
            log.warning(f"Camera p1 key is None for metadata. {self.metadata_path} .")
        
        p2 = safe_get(self.cam_ref, "p2", default=None)
        if not p2:
            log.warning(f"Camera p2 key is None for metadata. {self.metadata_path} .")

        camera_coefficients = {
            "f": f,
            "cx": cx,
            "cy": cy,
            "b1": b1,
            "b2": b2,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "p1": p1,
            "p2": p2
        }
        self.metadata["camera_info"]["camera_coefficients"] = camera_coefficients

    def _get_pixel_width_height(self) -> None:
        """Get the pixel width and height from the camera_info."""
        pixel_width = safe_get(self.cam_ref, "pixel_width", default=None)
        if pixel_width:
            self.metadata["camera_info"]["pixel_width"] = pixel_width
        else:
            log.error(f"Pixel width key is None for metadata. {self.metadata_path} .")
        
        pixel_height = safe_get(self.cam_ref, "pixel_height", default=None)
        if pixel_height:
            self.metadata["camera_info"]["pixel_height"] = pixel_height
        else:
            log.error(f"Pixel height key is None for metadata. {self.metadata_path} .")

    def _get_fov(self) -> None:
        """Get the fov from the camera_info."""
        if self.fov:
            if "top_left_x" in self.fov:
                tl_x = safe_get(self.fov, "top_left_x")
            else:
                log.error(f"Top left x key is None for metadata. {self.metadata_path} .")

            if "top_left_y" in self.fov:
                tl_y = safe_get(self.fov, "top_left_y")
            else:
                log.error(f"Top left y key is None for metadata. {self.metadata_path} .")
            
            if "bottom_left_x" in self.fov:
                bl_x = safe_get(self.fov, "bottom_left_x")
            else:
                log.error(f"Bottom left x key is None for metadata. {self.metadata_path} .")

            if "bottom_left_y" in self.fov:
                bl_y = safe_get(self.fov, "bottom_left_y")
            else:
                log.error(f"Bottom left y key is None for metadata. {self.metadata_path} .")

            if "top_right_x" in self.fov:
                tr_x = safe_get(self.fov, "top_right_x")
            else:
                log.error(f"Top right x key is None for metadata. {self.metadata_path} .")

            if "top_right_y" in self.fov:
                tr_y = safe_get(self.fov, "top_right_y")
            else:
                log.error(f"Top right y key is None for metadata. {self.metadata_path} .")

            if "bottom_right_x" in self.fov:
                br_x = safe_get(self.fov, "bottom_right_x")
            else:
                log.error(f"Bottom right x key is None for metadata. {self.metadata_path} .")
            
            if "bottom_right_y" in self.fov:
                br_y = safe_get(self.fov, "bottom_right_y")
            else:
                log.error(f"Bottom right y key is None for metadata. {self.metadata_path} .")

            tl_xy = [tl_x, tl_y]
            tr_xy = [tr_x, tr_y]
            br_xy = [br_x, br_y]
            bl_xy = [bl_x, bl_y]

            if "height" in self.fov:
                image_height_m = safe_get(self.fov, "height")
            else:
                log.error(f"Height key is None for metadata. {self.metadata_path} .")

            if "width" in self.fov:
                image_width_m = safe_get(self.fov, "width")
            else:
                log.error(f"Width key is None for metadata. {self.metadata_path} .")

            if image_height_m and image_width_m:
                fov_area_cm2 = image_width_m * image_height_m * 10000  # Convert m² to cm²
            else:
                fov_area_cm2 = None
                log.error(f"Fov area is None for metadata. {self.metadata_path} ." )

            fov = {
                "top_left_xy": tl_xy,
                "top_right_xy": tr_xy,
                "bottom_right_xy": br_xy,
                "bottom_left_xy": bl_xy,
                "height": image_height_m,
                "width": image_width_m,
                "fov_area_cm2": fov_area_cm2
            }
            self.metadata["camera_info"]["fov"] = fov
            
            log.debug(f"Fov key found in metadata: {self.fov}")
        else:
            fov = {
                "top_left_xy": None,
                "top_right_xy": None,
                "bottom_right_xy": None,
                "bottom_left_xy": None,
                "height": None,
                "width": None,
                "fov_area_cm2": None
            }
            self.metadata["camera_info"]["fov"] = fov
            log.error(f"Fov key is None for metadata. {self.metadata_path} .")
        


        
        
    ################################################################################################
    ######################################## Annotations ###########################################
    ################################################################################################

    def _get_annotations(self) -> None:
        """Get the annotations from the metadata."""
        if "annotations" in self.metadata:
            annotations = safe_get(self.metadata, "annotations")
            self.annotation_key_flag = "annotations"
            if annotations or annotations == []:
                self.metadata["annotations"] = annotations
                log.debug(f"annotations key found in metadata.")
            
        elif "bboxes" in self.metadata:
            annotations = safe_get(self.metadata, "bboxes")
            self.annotation_key_flag = "bboxes"
            if annotations or annotations == []:
                self.metadata["annotations"] = annotations
                log.debug(f"'bboxes' key found in metadata.")
        
        elif "annotation" in self.metadata and "bboxes" in self.metadata["annotation"]:
            annotations = safe_get(self.metadata, "annotations")
            bboxes = safe_get(self.metadata, "bboxes")
            combined_annotations = annotations + bboxes
            self.annotation_key_flag = "both_annotations"
            
            if annotations or annotations == []:
                self.metadata["both_annotations"] = combined_annotations
                log.debug(f"'bboxes' key found in metadata.")
        else:
            log.error(f"'annotations' or 'bboxes' key was not found in metadata. {self.metadata_path} .")

    
    def _clean_annotations(self) -> None:
        annot_cleaner = AnnotationCleaner(self.cfg, self.metadata_path)

        annotations = safe_get(self.metadata, self.annotation_key_flag, default=None)

        if annotations:
            for i, annotation in enumerate(annotations):
                cleaned_annotation = annot_cleaner.clean_annotations(annotation)
                keys_to_keep = self.full_schema['properties']['annotations']['items']['properties']
                cleaned_annotation = {key: cleaned_annotation[key] for key in keys_to_keep if key in cleaned_annotation}
                annotations[i] = cleaned_annotation
    
        
        self.class_ids = set(annot_cleaner.class_ids)
    
    ################################################################################################
    ######################################## Categories ############################################
    ################################################################################################
    
    def _get_categories(self) -> None:
        """Get the categories from the metadata."""
        
        if "categories" in self.metadata:
            self.categories_key_flag = "categories"
            categories = safe_get(self.metadata, "categories")
            if categories or categories == []:
                self.metadata["categories"] = categories
                log.debug(f"categories key found in metadata.")
        
        elif "cls" in self.metadata:
            self.categories_key_flag = "cls"
            categories = safe_get(self.metadata, "cls")
            if categories or categories == []:
                self.metadata["categories"] = categories
                log.debug(f"'cls' key found in metadata.")
        
        elif "category" in self.metadata:
            self.categories_key_flag = "category"
            categories = safe_get(self.metadata, "category")
            if categories or categories == []:
                self.metadata["categories"] = categories
                log.debug(f"'category' key found in metadata.")

        else:
            log.debug(f"'categories', 'cls', or 'category' key was not found in metadata. {self.metadata_path} . Creating empty 'categories' key.")
            self.categories_key_flag = "categories"
            self.metadata["categories"] = []

    
    def _clean_categories(self) -> None:
        category = self.species_info["species"]
        reindexed_categories = {}
        for _, value in category.items():
            class_id = value.get("class_id")
            if class_id is not None:
                reindexed_categories[class_id] = value
        cats = []
        for class_id in self.class_ids:
            if class_id in reindexed_categories:
                cat = reindexed_categories[class_id]
                
                if "collection_location" in cat:
                    cat.pop("collection_location")

                if "collection_timing" in cat:
                    cat.pop("collection_timing")
                
                cats.append(cat)
            else:
                log.error(f"Class id {class_id} not found in species_info. {self.metadata_path} .")
        
        self.metadata["categories"] = cats

    
    def _organize_exif_meta(self) -> None:
        """Organize the exif_meta keys in the metadata."""
        category_keys = self.full_schema['properties']['exif_meta']['required']
        metadata_keys = list(self.metadata['exif_meta'].keys())
        for key in metadata_keys:
            if key not in category_keys:
                self.metadata['exif_meta'].pop(key)

        # Order the self.metadata['exif_meta'] dictionary using the category_keys order
        ordered_category = {key: self.metadata['exif_meta'][key] for key in category_keys if key in self.metadata['exif_meta']}
        self.metadata['exif_meta'] = ordered_category

    def _organize_camera_info(self) -> None:
        """Organize the camera_info keys in the metadata."""
        category_keys = self.full_schema['properties']['camera_info']['required']
        metadata_keys = list(self.metadata['camera_info'].keys())
        for key in metadata_keys:
            if key not in category_keys:
                self.metadata['camera_info'].pop(key)

        # Order the self.metadata['camera_info'] dictionary using the category_keys order
        ordered_category = {key: self.metadata['camera_info'][key] for key in category_keys if key in self.metadata['camera_info']}
        self.metadata['camera_info'] = ordered_category

    def _organize_camera_info(self) -> None:
        """Organize the camera_info keys in the metadata."""
        category_keys = self.full_schema['properties']['camera_info']['required']
        metadata_keys = list(self.metadata['camera_info'].keys())
        for key in metadata_keys:
            if key not in category_keys:
                self.metadata['camera_info'].pop(key)

        # Order the self.metadata['camera_info'] dictionary using the category_keys order
        ordered_category = {key: self.metadata['camera_info'][key] for key in category_keys if key in self.metadata['camera_info']}
        self.metadata['camera_info'] = ordered_category

    def _organize_root_keys(self) -> None:
        """Organize the root keys in the metadata."""
        root_keys = self.full_schema['required']
        metadata_keys = list(self.metadata.keys())
        for key in metadata_keys:
            if key not in root_keys:
                self.metadata.pop(key)

        # Order the self.metadata dictionary using the root_key order
        ordered_metadata = {key: self.metadata[key] for key in root_keys if key in self.metadata}
        self.metadata = ordered_metadata

    
    def _replace_nan_with_null(self):
        import collections
        """Iteratively replace NaN with None in self.metadata."""
        stack = collections.deque([self.metadata])
        
        while stack:
            current = stack.pop()
            
            if isinstance(current, dict):
                for key, value in current.items():
                    if isinstance(value, float) and np.isnan(value):
                        current[key] = None
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
                        
            elif isinstance(current, list):
                for i in range(len(current)):
                    if isinstance(current[i], float) and np.isnan(current[i]):
                        current[i] = None
                    elif isinstance(current[i], (dict, list)):
                        stack.append(current[i])
    

    
        
    