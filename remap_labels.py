import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import shapefile

from bbox.bbox_transformations import BBoxFilter, BBoxMapper
from bbox.connectors import BBoxComponents, SfMComponents
from bbox.io_utils import ParseXML, ParseYOLOCsv
from semif_utils.utils import rescale_bbox

log = logging.getLogger(__name__)


class RemapLabels:

    def __init__(self, cfg: DictConfig) -> None:
        self.data_dir = Path(cfg.data.datadir)
        self.developed_dir = Path(cfg.data.developeddir)  # data_root
        self.batch_dir = Path(cfg.data.batchdir)
        self.autosfmdir = Path(cfg.autosfm.autosfmdir)
        self.metadata = self.batch_dir / "metadata"
        self.reference = self.autosfmdir / "reference"
        self.raw_label = self.autosfmdir / "detections.csv"
        self.downscaled = cfg.autosfm.autosfm_config.downscale.enabled

        if self.downscaled:
            self.image_dir = self.autosfmdir / "downscaled_photos"
        else:
            self.image_dir = self.batch_dir / "images"
        self.fullres_image_path = self.batch_dir / "images"

    @property
    def camera_reference(self):
        connector = SfMComponents(self.reference)
        gcp_reference = connector.gcp_reference
        cam_ref = connector.camera_reference
        return cam_ref

    def remap_labels(self):
        # Initialize the reader which will read the annotation files and convert the bounding
        # boxes to the desired format
        reader = ParseYOLOCsv(image_path=self.image_dir,
                              label_path=self.raw_label,
                              fullres_image_path=self.fullres_image_path)

        # Initialize the connector and get a list of all the images
        box_connector = BBoxComponents(
            self.data_dir,
            self.developed_dir,
            self.batch_dir,
            self.image_dir,
            self.camera_reference,
            reader,
            self.raw_label,
            fullres_image_path=self.fullres_image_path)
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
                    # box = rescale_bbox(box, 0)
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
            img.image_path = Path("images", Path(img.image_path).name)
            img.save_config(self.metadata)
        log.info("Saving complete.")
        return imgs


def main(cfg: DictConfig) -> None:
    rmpl = RemapLabels(cfg)
    imgs = rmpl.remap_labels()
