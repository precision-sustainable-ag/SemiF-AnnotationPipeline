import os
import subprocess
import glob

from omegaconf import DictConfig


def main(cfg: DictConfig) -> None:
    print("test")
        
    # Check to see if raw files exist in the folder
    raw_file_list = glob.glob(cfg.data.rawbatchdir + "*.ARW")
    if(len(raw_file_list)==0):
        assert False, "There appears to be no raw images in the specified directory"
        
    # Check to see if there are the same number of jpg and raw files (will hint at an incomplete upload)
    jpg_file_list = glob.glob(cfg.data.rawbatchdir + "*.JPG")
    if(not ( len(raw_file_list) == len(jpg_file_list))):
        assert False, "Data is likely missing from the upload folder"
    
    os.makedirs(cfg.data.batchdir, exist_ok = True)
    
    dev_profile = "2022-04-12_NC_indoor.pp3"
     
    
    # Compose the command
    exe_command = f"rawtherapee-cli \
    -o {cfg.data.batchdir} \
    -p {cfg.im_development.dev_profiles_dir}/{dev_profile} \
    -j99 \
    -c {cfg.data.rawbatchdir}"
    print(exe_command)
    try:
        # Run the autoSfM command
        subprocess.run(exe_command, shell=True, check=True)
    except Exception as e:
        raise e
