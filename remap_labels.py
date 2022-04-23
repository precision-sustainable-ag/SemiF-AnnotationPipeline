from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from bbox.bbox_transformations import BBoxFilter, BBoxMapper
from bbox.connectors import BBoxComponents, SfMComponents
from bbox.io_utils import ParseXML, ParseYOLO


class RemapLabels:

    def __init__(self, cfg: DictConfig) -> None:
        self.asfm_root = Path(cfg.autosfm.autosfmdir)
        self.reference_path = self.asfm_root / "reference"
        self.metadata = cfg.general.batchdir + "/labels"  #self.asfm_root / "metadata"

        self.image_dir = cfg.general.imagedir
        self.raw_label = self.asfm_root / "annotations"
        # self.raw_label = cfg.detect.detections_csv

    @property
    def camera_reference(self):
        connector = SfMComponents(self.reference_path)
        gcp_reference = connector.gcp_reference
        cam_ref = connector.camera_reference
        return cam_ref

    def remap_labels(self):
        # Initialize the reader which will read the annotation files and convert the bounding
        # boxes to the desired format
        reader = ParseXML(image_path=self.image_dir, label_path=self.raw_label)
        # Initialize the connector and get a list of all the images
        box_connector = BBoxComponents(self.camera_reference, reader,
                                       self.image_dir, self.raw_label)
        imgs = box_connector.images
        # Map the bounding boxes from local coordinates to global coordinate system
        mapper = BBoxMapper(imgs)
        mapper.map()
        # Sanity check
        for img in imgs:
            for box in img.bboxes:
                assert len(box._overlapping_bboxes) == 0
        # Apply "Non-Maxima Supression" to identify ideal boundng boxes.
        # Note that the operations are done in place, i.e. imgs, which is
        # a list of Image objects have BBox objects associated with them.
        # The properties of these objects are changed by the BBoxFilter.
        bbox_filter = BBoxFilter(imgs)
        bbox_filter.deduplicate_bboxes()
        # Save the config
        for img in imgs:
            Path(self.metadata).mkdir(parents=True, exist_ok=True)
            img.save_config(self.metadata)
        return imgs


def main(cfg: DictConfig) -> None:
    rmpl = RemapLabels(cfg)
    imgs = rmpl.remap_labels()
    # TODO save to database
