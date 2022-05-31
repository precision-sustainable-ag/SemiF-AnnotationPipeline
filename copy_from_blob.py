import shutil
from pathlib import Path

from omegaconf import DictConfig

def main(cfg: DictConfig) -> None:

    batch_id = cfg.general.batch_id
    developed_home = Path(cfg.data.developeddir, batch_id)
    developed_home.mkdir(exist_ok=False) # Ensure that the ID has not already been processed

    # Copy the developed images
    src = Path(cfg.blob_storage.developeddir, batch_id, "images")
    dst = Path(developed_home)
    shutil.copy(src, dst)

    # Copy the Ground Control Points
    src = Path(cfg.blob_storage.uploaddir, "GroundControlPoints.csv")
    dst = Path(cfg.data.developeddir)
    shutil.copy(src, dst)

    # Copy the model if not present
    local_model_path = Path(cfg.detect.model_path)
    if not local_model_path.exists():
        blob_model_path = Path(cfg.blob_storage.model_path, cfg.model.checkpoint_name)
        dst = Path(local_model_path)
        shutil.copy(blob_model_path, dst)
