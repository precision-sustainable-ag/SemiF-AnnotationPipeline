import os
import shutil
import re
from pathlib import Path

from omegaconf import DictConfig

def parse(filename):

    # Parse MD_Row-3_1655826853.JPG
    state, row, timestamp = re.findall("(MD|NC|TX)_Row-(\d+)_(\d+)", filename)[0]
    ext = filename.split(".")[-1]
    return state, int(row), int(timestamp), ext

def rename(state, row, pot, timestamp, ext):

    return f"{state}_{row}_{pot}_{timestamp}.0.{ext}"

def main(cfg: DictConfig) -> None:

    batch_id = cfg.general.batch_id
    developed_home = Path(cfg.data.batchdir)
    developed_home.mkdir(exist_ok=False) # Ensure that the ID has not already been processed

    # Copy the developed images
    src = Path(cfg.blob_storage.developeddir, batch_id, "images")
    dst = Path(developed_home, "images")
    shutil.copytree(src, dst)

    if cfg.data.rename:
        files = os.listdir(dst)
        rowmap = dict()
        for file in files:
            state, row, timestamp, ext = parse(file)
            row_files = rowmap.get(row, [])
            row_files.append((file, state, timestamp, ext))
            rowmap[row] = row_files
        
        # Sort
        for row, row_files in rowmap.items():

            # Sort according to timestamp
            row_files.sort(key=lambda x: x[2])
            # Rename
            for idx, file in enumerate(row_files):
                org_file, state, timestamp, ext = file
                renamed_file = rename(state, row, idx+1, timestamp, ext)

                _src = Path(dst, org_file)
                _dst = Path(dst, renamed_file)
                shutil.move(_src, _dst)

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
