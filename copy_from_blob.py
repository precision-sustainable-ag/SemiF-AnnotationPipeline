import shutil
from pathlib import Path

from omegaconf import DictConfig

def main(cfg: DictConfig) -> None:

    batch_id = cfg.general.batch_id
    developed_home = Path(cfg.data.batchdir)
    developed_home.mkdir(exist_ok=False) # Ensure that the ID has not already been processed

    # Copy the developed images
    src = Path(cfg.blob_storage.developeddir, batch_id, "images")
    dst = Path(developed_home, "images")
    shutil.copytree(src, dst)

    # Copy the Ground Control Points
    src = Path(cfg.blob_storage.uploaddir, batch_id, "GroundControlPoints.csv")
    dst = Path(cfg.data.uploaddir, batch_id)
    dst.mkdir(exist_ok=False)
    dst = Path(dst, "GroundControlPoints.csv")
    shutil.copy(src, dst)

    # Copy the detection model if not present
    local_model_path = Path(cfg.detect.model_path)
    if not local_model_path.exists():
        blob_model_path = Path(cfg.blob_storage.modeldir, "plant_detector", cfg.detect.model_filename)
        dst = Path(local_model_path)
        shutil.copy(blob_model_path, dst)

    # Copy the shapefiles
    location = batch_id.split("_")[0]
    shapefile_path = Path(cfg.data.utilsdir, location, "shapefiles")
    if not shapefile_path.exists():
        src = Path(cfg.blob_storage.utilsdir, location, "shapefiles")
        assert src.exists()
        dst = shapefile_path
        shutil.copytree(src, dst)