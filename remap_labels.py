from pathlib import Path
import logging

from omegaconf import DictConfig
import hydra
from tqdm import tqdm

from bbox.bbox_transformations import BBoxFilter, BBoxMapper
from bbox.connectors import BBoxComponents, SfMComponents
from bbox.io_utils import ParseXML, ParseYOLOCsv

log = logging.getLogger(__name__)


class RemapLabels:

    def __init__(self, cfg: DictConfig) -> None:
        self.asfm_root = Path(cfg.autosfm.autosfmdir)
        self.reference_path = self.asfm_root / "reference"
        self.metadata = cfg.general.batchdir + "/labels"  #self.asfm_root / "metadata"
        
        if cfg.autosfm.autosfm_config.downscale.enabled:
            self.image_dir = Path(cfg.general.batchdir, "autosfm", "downscaled_photos")
        else:
            self.image_dir = cfg.general.imagedir
        self.raw_label = cfg.detect.detections_csv

    @property
    def camera_reference(self):
        connector = SfMComponents(self.reference_path)
        gcp_reference = connector.gcp_reference
        cam_ref = connector.camera_reference
        return cam_ref

    def remap_labels(self):
        # Initialize the reader which will read the annotation files and convert the bounding
        # boxes to the desired format
        reader = ParseYOLOCsv(image_path=self.image_dir,
                              label_path=self.raw_label)
        # Initialize the connector and get a list of all the images
        box_connector = BBoxComponents(self.camera_reference, reader,
                                       self.image_dir, self.raw_label)
        log.info("Fetching image metadata.")
        imgs = box_connector.images
        # Map the bounding boxes from local coordinates to global coordinate system
        log.info("Staring mapping.")
        mapper = BBoxMapper(imgs)
        mapper.map()
        # Sanity check
        for img in imgs:
            for box in img.bboxes:
                try:
                    assert len(box._overlapping_bboxes) == 0
                except AssertionError as e:
                    log.debug("Mapping failed> Reason: {}".format(str(e)))
                    raise e
        log.info("Mapping complete.")
        # Apply "Non-Maxima Supression" to identify ideal boundng boxes.
        # Note that the operations are done in place, i.e. imgs, which is
        # a list of Image objects have BBox objects associated with them.
        # The properties of these objects are changed by the BBoxFilter.
        log.info("Deduplicating bounding boxes.")
        bbox_filter = BBoxFilter(imgs)
        bbox_filter.deduplicate_bboxes()
        log.info("Deduplication complete.")
        log.info("Saving bounding box metadata.")
        # Save the config
        for img in imgs:
            Path(self.metadata).mkdir(parents=True, exist_ok=True)
            img.save_config(self.metadata)
        log.info("Saving complete.")
        return imgs


def main(cfg: DictConfig) -> None:
    rmpl = RemapLabels(cfg)
    imgs = rmpl.remap_labels()
    # TODO save to database
