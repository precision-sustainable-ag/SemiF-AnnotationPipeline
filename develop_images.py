import os
import subprocess
import glob
from pathlib import Path
from omegaconf import DictConfig


def main(cfg: DictConfig) -> None:
    print("test")
        
    # Check to see if raw files exist in the folder
    raw_file_list = glob.glob(cfg.blob_storage.rawbatchdir + "*.ARW")
    if(len(raw_file_list)==0):
        assert False, "There appears to be no raw images in the specified directory"
        
    # Check to see if there are the same number of jpg and raw files (will hint at an incomplete upload)
    jpg_file_list = glob.glob(cfg.blob_storage.rawbatchdir + "*.JPG")
    if(not ( len(raw_file_list) == len(jpg_file_list))):
        assert False, "Data is likely missing from the upload folder"
    
    output_path = Path(cfg.data.batchdir, "images")
    os.makedirs(output_path, exist_ok = True)
    
    dev_profile = "2022-04-12_NC_indoor.pp3"
     
    
    # Compose the command
    exe_command = f"rawtherapee-cli \
    -o {output_path} \
    -p {cfg.im_development.dev_profiles_dir}/{dev_profile} \
    -j99 \
    -c {cfg.blob_storage.rawbatchdir}"
    print(exe_command)
    try:
        # Run the autoSfM command
        subprocess.run(exe_command, shell=True, check=True)
    except Exception as e:
        raise e
