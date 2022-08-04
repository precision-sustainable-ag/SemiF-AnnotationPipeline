# SemiField Annotation Pipeline


<br>

# TODOs

## Environment Setup in Azure VM without GPU:

```Bash
conda update -n base conda
conda env create -f environment_nogpu.yml
conda activate semif_nogpu
conda install pytorch torchvision cpuonly -c pytorch
pip install /home/azureuser/autoSfM/autoSfM/package/dependencies/Metashape-1.8.0-cp35.cp36.cp37.cp38-abi3-linux_x86_64.whl
export agisoft_LICENSE="/home/azureuser/SemiF-AnnotationPipeline/autosfm/volumes/metashape/metashape-pro/metashape.lic"
```
<br>

# Environment Setup

Use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

Clone repo and create environment with [`environment.yml`](environment.yml) in [**Python>=3.9**](https://www.python.org/).

<br>

```
conda env create --file=environment.yml
```

<br>


# Run the pipeline


`SEMIF.py` runs the pipeline based on `conf/config.yaml`.
<br>

Settings in `conf/config.yaml` can be modified straight from the command line. 

```
python SEMIF.py general.batch_id=<batch_id> \
                general.task=segment_vegetation \
                general.vi=exg \
                general.clear_border=True \
                autosfm.metashape_key=<metashape_key>
```

Note that the ```metashape_key``` isn't currently required, and hence can be replaced with any random string. The key will be required after August 2022.

The ```general.batch_id``` and ```autosfm.metashape_key``` are mandatory fields.

<br>

# Data Overview

The WIR is made up of four main datasets, and a temporary upload container. More information about the dataset and structure can be found in the main data folder [blob_container](blob_container/README.md)

<p align="center">
<img src="Assets/overview.png" width=75%/>
</p>


<br>
<br>

# ```conf/config.yml``` for controlling execution
The ```conf/config.yml``` is used for controlling the execution of the pipelne. The ```multitask``` field enables sequential execution of the tasks listed in ```multitasks```. Note that the names of the tasks are sensitive to the order of execution and the names. The names of the tasks must be from ```test_mongodb```, ```segment_vegetation```, ```localize_plants```, ```remap_labels```, ```auto_sfm```. The order of exeqution must be as follows:

```auto_sfm```, ```localize_plants```, ```remap_labels```, ```segment_vegetation```

<br>

# AutoSfM pipeline triggering
```auto_sfm``` should be triggeretd and completed before ```remap_labels```. AutoSfM ([https://github.com/precision-sustainable-ag/autoSfM](https://github.com/precision-sustainable-ag/autoSfM)) is run using a Docker container, and needs a config file through which the execution is controlled. This config file is generated dynamically through the ```conf/config.yml``` file using the ```autosfm.autosfm_config``` field. This ensures that the same batch ID is used for processing both the pipelines.

Following directories from the repository are used for autoSfM:
1. ```autosfm/volumes```: This is the directory which contains the config.yml file used for controlling the execution, and the Metashape installation. The Metashape software has to be copied maually. The directiry structure of ```autosfm/volumes``` is fixed and should not be changed.
2. ```autosfm/exports```: The products of the autoSfM are stored here. This directory is mounted onto the Docker container at ```/home/psi_docker/autoSfM/exports```.
3. ```data/trials```: The data to be processed by autoSfM are stored here (this directory is shared with the rest of the pipeline). This directory is mounted on the Docker container at ```/home/psi_docker/autoSfM/storage```.

The flow of execution is as follows:
1. All the directories are mounted on the Docker container and the pipeline is executed.
2. The exports are moved to ```data/trial/{batch_id}/autosfm```.
3. The generated ```config.yml``` is copied to ```logs/autosfm/config_{batch_id}.yml``` for logging purposes.

Following directories from the repository are used for autoSfM:
1. ```autosfm/volumes```: This is the directory which contains the config.yml file used for controlling the execution, and the Metashape installation. The Metashape software has to be copied maually. The directiry structure of ```autosfm/volumes``` is fixed and should not be changed.
2. ```autosfm/exports```: The products of the autoSfM are stored here. This directory is mounted onto the Docker container at ```/home/psi_docker/autoSfM/exports```.
3. ```data/trials```: The data to be processed by autoSfM are stored here (this directory is shared with the rest of the pipeline). This directory is mounted on the Docker container at ```/home/psi_docker/autoSfM/storage```.

<br>

# Global bbox mapping with AutoSfM

More info can be found [**here**](bbox/README.md)

