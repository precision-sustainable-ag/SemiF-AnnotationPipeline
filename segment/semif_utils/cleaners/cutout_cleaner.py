import shutil
import logging
from pathlib import Path
from pprint import pprint
from omegaconf import DictConfig

from semif_utils.utils import (
    calculate_bbox_area_cm2, 
    match_season_to_date_ranges, 
    match_single_date_range_to_season, 
    read_json_file, 
    find_date_range_season
    )

log = logging.getLogger(__name__)
    
class CutoutMetadataCleaner:
    """Class to clean and reformat metadata files systematically."""
    
    def __init__(self, cfg: DictConfig, batch_path: Path):
        """Initialize the cleaner with the metadata to be cleaned."""
        self.cfg = cfg
        self.batch_path = batch_path

        self.species_info = read_json_file(cfg.data.species)
        self.cutout_schema = read_json_file(Path(cfg.data.utilsdir, "species_information", "cutout_schema.json"))
        self.date_ranges = cfg.date_ranges

        self.admin_classes = ["colorchecker", "unknown", "background"]
        
        self.pipeline = [
            # Root properties
            self._get_season,
            self._get_datetime,
            self._get_batch_id,
            self._get_image_id,
            self._get_cutout_id,
            self._get_cutout_number,
            self._get_cutout_height_width,
            self._get_lens_model,
            self._is_validated,
            self._cutout_version,
            self._add_bbot_version_number,
                        
            # Cutout properties
            self._get_cutout_props,
            self._get_cutout_props_bbox_area_cm2,
            self._get_blur_effect,
            self._get_num_components,
            self._get_cropout_rgb_mean,
            self._get_cropout_rgb_std,
            self._get_is_primary,
            self._get_extends_border,
            
            # Category properties
            self._get_category,
            self._get_class_id,
            self._get_usda_symbol,
            self._get_EPPO_code,
            self._get_group,
            self._get_class,
            self._get_subclass,
            self._get_order,
            self._get_family,
            self._get_genus,
            self._get_species,
            self._get_common_name,
            self._get_authority,
            self._get_growth_habit,
            self._get_duration,
            self._get_category_category,
            self._get_multi_species,
            self._get_link,
            self._get_note,
            self._get_hex_color,
            self._get_rgb_color,

            # Organize properties
            self._organize_cutout_props,
            self._organize_category,
            self._organize_root_keys,

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
        
        # Important
        self.metadata = metadata
        self.metadata_path = metadata_path
        
        # Being replaced with cached_data
        self.exif_meta = image_data.get('exif_meta')
        self.fov = image_data.get('fov')

        if "category" in self.metadata:
            self.common_name = self.metadata["category"].get("common_name")
        elif "categories"   in self.metadata:
            self.common_name = self.metadata["categories"].get("common_name")
        elif "cls" in self.metadata:
            self.common_name = self.metadata["cls"].get("common_name")
        else:
            self.common_name = None
        
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

    def _get_season(self) -> None:
        season = self.metadata.get('season')
        
        if not season or season.lower() in self.admin_classes:
            state_id = self.metadata.get('batch_id').split("_")[0]
            batch_date = self.metadata.get('batch_id').split("_")[1]
            date_range_season = find_date_range_season(batch_date, state_id, self.date_ranges)            
            season = match_single_date_range_to_season(date_range_season, self.possible_seasons)

        if season:
            self.metadata['season'] = season
            log.debug(f"Season key found in metadata: {season}")
        
        else:
            log.error(f"Season key not found in metadata. {self.metadata_path} .")

    def _get_datetime(self) -> None:
        datetime = self.metadata.get('datetime')
        if datetime:
            self.metadata['datetime'] = datetime
            log.debug(f"datetime key found in metadata: {datetime}")
        else:
            log.error(f"datetime key not found in metadata. {self.metadata_path} .")

    def _get_batch_id(self) -> None:
        batch_id = self.metadata.get('batch_id')
        if batch_id:
            self.metadata['batch_id'] = batch_id
            log.debug(f"batch_id key found in metadata: {batch_id}")
        else:
            log.error(f"batch_id key not found in metadata. {self.metadata_path} .")
    
    def _get_image_id(self) -> None:
        image_id = self.metadata.get('image_id')
        if image_id:
            self.metadata['image_id'] = image_id
            log.debug(f"image_id key found in metadata: {image_id}")
        else:
            log.error(f"image_id key not found in metadata. {self.metadata_path} .")

    def _get_cutout_id(self) -> None:
        cutout_id = self.metadata.get('cutout_id')
        if cutout_id:
            self.metadata['cutout_id'] = cutout_id
            log.debug(f"cutout_id key found in metadata: {cutout_id}")
        else:
            log.error(f"cutout_id key not found in metadata. {self.metadata_path} .")

    def _get_cutout_number(self) -> None:
        cutout_number = int(self.metadata.get('cutout_num'))
        
        if cutout_number:
            self.metadata['cutout_num'] = int(cutout_number)
            log.debug(f"cutout_num key found in metadata: {cutout_number}")

        elif cutout_number == 0:
            self.metadata['cutout_num'] = 0
            log.debug(f"cutout_num key found in metadata and is zero: {cutout_number}")
        
        else:
            log.error(f"cutout_num ({cutout_number}) key not found in metadata. {self.metadata_path} .")

    def _get_cutout_height_width(self) -> None:
        cutout_height = self.metadata.get('cutout_height')
        cutout_width = self.metadata.get('cutout_width')
        hwc = self.metadata.get('hwc')
        
        if cutout_height and cutout_width:
            self.metadata['cutout_height'] = int(cutout_height)
            self.metadata['cutout_width'] = int(cutout_width)
            log.debug(f"cutout_height and cutout_width keys found in metadata: {cutout_height}, {cutout_width}")
        
        elif hwc:
            self.metadata['cutout_height'] = int(hwc[0])
            self.metadata['cutout_width'] = int(hwc[1])
            log.debug(f"hwc key found in metadata: {hwc}")

        else:
            log.error(f"cutout_height and cutout_width keys not found in metadata. {self.metadata_path} .")

    def _get_lens_model(self) -> None:
        """Retrieve the lens model from the metadata or infer it."""
        lens_model = self.exif_meta.get('LensModel')
        if lens_model:
            self.metadata['lens_model'] = lens_model
            log.debug(f"lens_model key found in metadata: {lens_model}")
        else:
            log.error(f"lens_model key not found in metadata. {self.metadata_path} .")   

    def _add_bbot_version_number(self) -> None:
        """Add a version number to the metadata."""
        state_id = self.metadata['batch_id'].split("_")[0]
        season = self.metadata['season']
        date_range = self.date_ranges.get(state_id)
        date_range_season = match_season_to_date_ranges(season, date_range.keys())
        bbot_version = self.date_ranges[state_id][date_range_season]['bbot_version']
        
        if bbot_version:
            self.metadata['bbot_version'] = bbot_version
            log.debug(f"bbot_version key added to metadata: {bbot_version}")
        
        else:
            log.error(f"bbot_version key not found in metadata. {self.metadata_path} .")
     
#########################################################################################
################################### CUTOUT PROPS ########################################
#########################################################################################

    def _get_cutout_props(self) -> None:
        cutout_props = self.metadata.get('cutout_props')
        if cutout_props:
            self.metadata['cutout_props'] = cutout_props
            log.debug(f"cutout_props key found in metadata. Has {len(cutout_props.keys())} keys")
        else:
            log.error(f"cutout_props key not found in metadata. {self.metadata_path} .")

    def _get_cutout_props_bbox_area_cm2(self) -> None:
        if self.fov: 
            cutout_width = self.metadata.get('cutout_width')
            cutout_height = self.metadata.get('cutout_height')
            
            fullres_width = self.exif_meta.get('ImageWidth')
            fullres_height = self.exif_meta.get('ImageLength')
            
            image_height_m, image_width_m = self.fov.get('height'), self.fov.get('width')
            bbox_area_cm2 = calculate_bbox_area_cm2(image_height_m, image_width_m, cutout_height, cutout_width, fullres_width, fullres_height)

            self.metadata['cutout_props']['bbox_area_cm2'] = bbox_area_cm2
                
        else:
            log.error(f"fov not present. Could not calculate bbox_area_cm2 for {self.metadata_path} .")

            
    def _get_blur_effect(self) -> None:
        """Retrieve the blur effect from the metadata or infer it."""
        blur_effect = self.metadata['cutout_props'].get('blur_effect')
        
        if blur_effect:
            self.metadata['cutout_props']['blur_effect'] = blur_effect
            log.debug(f"blur_effect key found in metadata: {blur_effect}")
        
        elif (blur_effect == 0 or blur_effect is None) and self.common_name.lower() in self.admin_classes:
            self.metadata['cutout_props']['blur_effect'] = None
            log.debug(f"blur_effect key found in metadata and is zero: {blur_effect}")

        else:
            log.error(f"blur_effect {blur_effect} key not found in metadata. Common name: {self.common_name}. Path: {self.metadata_path} .")
            # real_path = str(self.metadata_path).replace('data/working', '/mnt/research-projects/s/screberg/').replace(".json", ".jpg")
            # blur_effect_dir = Path("blur_effect_not_found")
            # blur_effect_dir.mkdir(exist_ok=True)
            # shutil.copy(real_path, blur_effect_dir)
            # log.error(f"blur_effect {blur_effect} key not found in metadata. Setting it to None. Likely no cutout segment was generated. Common name: {self.common_name}. Path: {self.metadata_path} .")
            self.metadata['cutout_props']['blur_effect'] = None

    def _get_num_components(self) -> None:
        """Retrieve the number of components from the metadata or infer it."""
        num_components = self.metadata['cutout_props'].get('num_components')
        
        if num_components:
            self.metadata['cutout_props']['num_components'] = num_components
            log.debug(f"num_components key found in metadata: {num_components}")
        
        elif num_components == 0 and self.common_name.lower() == "colorchecker":
            self.metadata['cutout_props']['num_components'] = num_components
            log.debug(f"num_components key found in metadata and is zero: {num_components}")
        
        else:
            log.error(f"num_components key not found in metadata. Setting it to None. Likely no cutout segment was generated. Common name: {self.common_name}. Path: {self.metadata_path} .")
            # real_path = str(self.metadata_path).replace('data/working', '/mnt/research-projects/s/screberg/').replace(".json", ".png")
            # num_components_dir = Path("num_components_not_found")
            # num_components_dir.mkdir(exist_ok=True)
            # shutil.copy(real_path, num_components_dir)
            self.metadata['cutout_props']['num_components'] = None
    
    def _get_cropout_rgb_mean(self) -> None:
        cropout_rgb_mean = self.metadata['cutout_props'].get('cropout_rgb_mean')

        if not cropout_rgb_mean:
            if "cropout_descriptive_stats" in self.metadata['cutout_props']:
                r_mean = self.metadata['cutout_props']['cropout_descriptive_stats']['channel_r'].get('mean')
                g_mean = self.metadata['cutout_props']['cropout_descriptive_stats']['channel_g'].get('mean')
                b_mean = self.metadata['cutout_props']['cropout_descriptive_stats']['channel_b'].get('mean')
                cropout_rgb_mean = [r_mean, g_mean, b_mean]
                log.debug(f"cropout_rgb_mean key found in cropout_descriptive_stats: {cropout_rgb_mean}")

            elif "cropout_rgb_mean" in self.metadata['cutout_props']:
                cropout_rgb_mean = self.metadata['cutout_props']['cropout_rgb_mean']
                log.debug(f"cropout_rgb_mean  found in cropout_rgb_mean: {cropout_rgb_mean}")

            else:
                log.error(f"cropout_rgb_mean key not found in metadata. {self.metadata_path} .")    
        
        if cropout_rgb_mean:
            cropout_rgb_mean = [x/255.0 for x in cropout_rgb_mean]
            log.debug(f"cropout_rgb_mean key found in metadata: {cropout_rgb_mean}")
            self.metadata['cutout_props']['cropout_rgb_mean'] = cropout_rgb_mean

    def _get_cropout_rgb_std(self) -> None:
        cropout_rgb_std = self.metadata['cutout_props'].get('cropout_rgb_std')
        
        if not cropout_rgb_std:
            if "cropout_descriptive_stats" in self.metadata['cutout_props']:
                r_std = self.metadata['cutout_props']['cropout_descriptive_stats']['channel_r'].get('std')
                g_std = self.metadata['cutout_props']['cropout_descriptive_stats']['channel_g'].get('std')
                b_std = self.metadata['cutout_props']['cropout_descriptive_stats']['channel_b'].get('std')
                cropout_rgb_std = [r_std, g_std, b_std]
                log.debug(f"cropout_rgb_std key found in cropout_descriptive_stats: {cropout_rgb_std}")

            elif "cropout_rgb_std" in self.metadata['cutout_props']:
                cropout_rgb_std = self.metadata['cutout_props']['cropout_rgb_std']
                log.debug(f"cropout_rgb_std  found in cropout_rgb_std: {cropout_rgb_std}")

            else:
                log.error(f"cropout_rgb_std key not found in metadata. {self.metadata_path} .")    
        
        if cropout_rgb_std:
            cropout_rgb_std = [x/255.0 for x in cropout_rgb_std]
            log.debug(f"cropout_rgb_std key found in metadata: {cropout_rgb_std}")
            self.metadata['cutout_props']['cropout_rgb_std'] = cropout_rgb_std

    def _get_is_primary(self) -> None:
        """Retrieve the is_primary field from the metadata."""
        is_primary = self.metadata.get('is_primary', None)
        if is_primary is None:

            is_primary = self.metadata['cutout_props'].get('is_primary')
        
        if is_primary == True or is_primary == False:
            self.metadata['cutout_props']['is_primary'] = is_primary
            log.debug(f"is_primary key found in metadata: {is_primary}")
        
        else:
            log.error(f"is_primary key not found in metadata. {self.metadata_path} .")

    def _get_extends_border(self) -> None:
        """Retrieve the is_primary field from the metadata."""
        extends_border = self.metadata.get('extends_border', None)
        if extends_border is None:

            extends_border = self.metadata['cutout_props'].get('extends_border')
        
        if extends_border == True or extends_border == False:
            self.metadata['cutout_props']['extends_border'] = extends_border
            log.debug(f"extends_border key found in metadata: {extends_border}")
        
        else:
            log.error(f"extends_border key not found in metadata. {self.metadata_path} .")

#####################################################################################
################################### CATEGORY ########################################
#####################################################################################

    def _get_category(self) -> None:
        """Retrieve the category field from the metadata."""
        category = self.metadata.get('category')
        
        if not category:
            category = self.metadata.get('cls')
            log.debug(f"cls key found in metadata: {category['common_name']}")

            if not category:
                category = self.metadata.get('categories')
                log.debug(f"categories key found in metadata: {category['common_name']}")

        if category:
            self.metadata['category'] = category
            log.debug(f"category key found in metadata: {category}")
        else:
            log.error(f"category key not found in metadata. {self.metadata_path} .")
    
    def _get_class_id(self) -> None:
        """Retrieve the class_id field from the metadata."""
        class_id = self.metadata['category'].get('class_id')
        
        if class_id:
            self.metadata['category']['class_id'] = class_id
            log.debug(f"class_id key found in metadata: {class_id}")
        
        else:
            log.error(f"class_id key not found in metadata. {self.metadata_path} .")

    def _get_usda_symbol(self) -> None:
        """Retrieve the usda_symbol field from the metadata."""
        usda_symbol = self.metadata['category'].get('USDA_symbol')
        cname = self.common_name
        if usda_symbol:
            if usda_symbol.lower() in self.admin_classes:
                usda_symbol = cname.lower()
            else:
                usda_symbol = usda_symbol.upper()
            self.metadata['category']['USDA_symbol'] = usda_symbol
            log.debug(f"usda_symbol key found in metadata: {usda_symbol}")
        
        else:
            log.error(f"usda_symbol key not found in metadata. {self.metadata_path} .")

    def _get_EPPO_code(self) -> None:
        """Retrieve the EPPO_code field from the metadata."""
        EPPO_code = self.metadata['category'].get('EPPO')
        
        if EPPO_code:
            self.metadata['category']['EPPO'] = EPPO_code.upper()
            log.debug(f"EPPO key found in metadata: {EPPO_code}")
        
        else:
            log.error(f"EPPO key not found in metadata. {self.metadata_path} .")

    def _get_group(self) -> None:
        """Retrieve the group field from the metadata."""
        group = self.metadata['category'].get('group')
        
        if group:
            self.metadata['category']['group'] = group.lower()
            log.debug(f"group key found in metadata: {group}")
        
        else:
            log.error(f"group key not found in metadata. {self.metadata_path} .")

    def _get_class(self) -> None:
        """Retrieve the class field from the metadata."""
        class_ = self.metadata['category'].get('class')
        
        if class_:
            self.metadata['category']['class'] = class_.title()
            log.debug(f"class key found in metadata: {class_}")
        
        else:
            log.error(f"class key not found in metadata. {self.metadata_path} .")

    def _get_subclass(self) -> None:
        """Retrieve the subclass field from the metadata."""
        subclass = self.metadata['category'].get('subclass')
        
        if subclass:
            self.metadata['category']['subclass'] = subclass.title()
            log.debug(f"subclass key found in metadata: {subclass}")
        
        else:
            log.error(f"subclass key not found in metadata. {self.metadata_path} .")

    def _get_order(self) -> None:
        """Retrieve the order field from the metadata."""
        order = self.metadata['category'].get('order')
        
        if order:
            self.metadata['category']['order'] = order.title()
            log.debug(f"order key found in metadata: {order}")
        
        else:
            log.error(f"order key not found in metadata. {self.metadata_path} .")

    def _get_family(self) -> None:
        """Retrieve the family field from the metadata."""
        family = self.metadata['category'].get('family')
        
        if family:
            self.metadata['category']['family'] = family.title()
            log.debug(f"family key found in metadata: {family}")
        
        else:
            log.error(f"family key not found in metadata. {self.metadata_path} .")

    def _get_genus(self) -> None:
        """Retrieve the genus field from the metadata."""
        genus = self.metadata['category'].get('genus')
        
        if genus:
            self.metadata['category']['genus'] = genus.title()
            log.debug(f"genus key found in metadata: {genus}")
        
        else:
            log.error(f"genus key not found in metadata. {self.metadata_path} .")

    def _get_species(self) -> None:
        """Retrieve the species field from the metadata."""
        species = self.metadata['category'].get('species')
        
        if species:
            self.metadata['category']['species'] = species.title()
            log.debug(f"species key found in metadata: {species}")
        
        else:
            log.error(f"species key not found in metadata. {self.metadata_path} .")

    def _get_common_name(self) -> None:
        """Retrieve the common_name field from the metadata."""
        common_name = self.metadata['category'].get('common_name')
        
        if common_name:
            self.metadata['category']['common_name'] = common_name.title()
            log.debug(f"common_name key found in metadata: {common_name}")
        
        else:
            log.error(f"common_name key not found in metadata. {self.metadata_path} .")

    def _get_authority(self) -> None:
        """Retrieve the authority field from the metadata."""
        authority = self.metadata['category'].get('authority')
        
        if authority:
            self.metadata['category']['authority'] = authority.title()
            log.debug(f"authority key found in metadata: {authority}")
        
        else:
            log.error(f"authority key not found in metadata. {self.metadata_path} .")

    def _get_growth_habit(self) -> None:
        """Retrieve the growth_habit field from the metadata."""
        growth_habit = self.metadata['category'].get('growth_habit')
        
        if growth_habit:
            self.metadata['category']['growth_habit'] = growth_habit.lower()
            log.debug(f"growth_habit key found in metadata: {growth_habit}")
        
        else:
            log.error(f"growth_habit key not found in metadata. {self.metadata_path} .")

    def _get_duration(self) -> None:
        """Retrieve the duration field from the metadata."""
        duration = self.metadata['category'].get('duration')
        
        if duration:
            self.metadata['category']['duration'] = duration.lower()
            log.debug(f"duration key found in metadata: {duration}")
        
        else:
            log.error(f"duration key not found in metadata. {self.metadata_path} .")

    def _get_category_category(self) -> None:
        """Retrieve the category field from the metadata."""
        category_category = self.metadata['category'].get('category')
        
        cname = self.common_name
        if not category_category:
            USDA_symbol = self.metadata['category'].get('USDA_symbol')
            
            if USDA_symbol.lower() in self.admin_classes:
                category_category = cname.lower()
            
            else:
                species_dict = self.species_info['species'][USDA_symbol]
                category_category = species_dict.get('category')
            
        
        if category_category:
            log.debug(f"category_category key found in metadata: {category_category}")
            self.metadata['category']['category'] = category_category.lower()
        
        else:
            log.error(f"category_category key not found in metadata. {self.metadata_path} .")

    def _get_multi_species(self) -> None:
        """Retrieve the multi_species field from the metadata."""
        
        multi_species = self.metadata['category'].get('multi_species_USDA_symbol', None)       
        self.metadata['category']['multi_species_USDA_symbol'] = multi_species
        log.debug(f"multi_species_USDA_symbol key found in metadata: {multi_species}")

    def _get_link(self) -> None:
        """Retrieve the link field from the metadata."""
        link = self.metadata['category'].get('link')
        cname = self.common_name
        if link:
            self.metadata['category']['link'] = link
            log.debug(f"link key found in metadata: {link}")
        
        elif not link and (cname in self.admin_classes):
            self.metadata['category']['link'] = ""
            log.debug(f"link key found in metadata and is None: {link} because common_name is {self.common_name}")

        elif not link and (cname not in self.admin_classes):
            species_dict = self.species_info['species'][self.metadata['category']['USDA_symbol']]
            link = species_dict.get('link')
            self.metadata['category']['link'] = link
            log.debug(f"link key found in metadata and is None: {link} because common_name is {self.common_name}")

        else:
            log.error(f"link key not found in metadata category. {self.metadata_path} .")

    def _get_note(self) -> None:
        """Retrieve the notes field from the metadata."""
        note = self.metadata['category'].get('note')
        self.metadata['category']['note'] = note
        log.debug(f"note key found in metadata: {note}")

    def _get_hex_color(self) -> None:
        hex = self.metadata['category'].get('hex')
        if hex:
            self.metadata['category']['hex'] = hex
            log.debug(f"hex key found in metadata: {hex}")
        else:
            log.error(f"hex key not found in metadata. {self.metadata_path} .")

    def _get_rgb_color(self) -> None:
        rgb = self.metadata['category'].get('rgb')
        if rgb:
            self.metadata['category']['rgb'] = rgb
            log.debug(f"rgb key found in metadata: {rgb}")
        else:
            log.error(f"rgb key not found in metadata. {self.metadata_path} .")

#####################################################################################
################################### ORGANIZE ########################################
#####################################################################################

    def _organize_cutout_props(self) -> None:
        """Organize the cutout_props keys in the metadata."""
        cutout_props_keys = self.cutout_schema['properties']['cutout_props']['required']
        metadata_keys = list(self.metadata['cutout_props'].keys())
        for key in metadata_keys:
            if key not in cutout_props_keys:
                self.metadata['cutout_props'].pop(key)

        # Order the self.metadata['category'] dictionary using the category_keys order
        ordered_cutout_props = {key: self.metadata['cutout_props'][key] for key in cutout_props_keys if key in self.metadata['cutout_props']}
        self.metadata['cutout_props'] = ordered_cutout_props

    def _organize_category(self) -> None:
        """Organize the category keys in the metadata."""
        category_keys = self.cutout_schema['properties']['category']['required']
        metadata_keys = list(self.metadata['category'].keys())
        for key in metadata_keys:
            if key not in category_keys:
                self.metadata['category'].pop(key)

        # Order the self.metadata['category'] dictionary using the category_keys order
        ordered_category = {key: self.metadata['category'][key] for key in category_keys if key in self.metadata['category']}
        self.metadata['category'] = ordered_category

    def _is_validated(self) -> None:
        """Check if the metadata is valid."""
        self.metadata['validated'] = False

    def _cutout_version(self) -> None:
        version = self.cfg.convert.versions.cutouts
        self.metadata['cutout_version'] = version

    def _organize_root_keys(self) -> None:
        """Organize the root keys in the metadata."""
        root_keys = self.cutout_schema['required']
        metadata_keys = list(self.metadata.keys())
        for key in metadata_keys:
            if key not in root_keys:
                self.metadata.pop(key)

        # Order the self.metadata dictionary using the root_key order
        ordered_metadata = {key: self.metadata[key] for key in root_keys if key in self.metadata}
        self.metadata = ordered_metadata