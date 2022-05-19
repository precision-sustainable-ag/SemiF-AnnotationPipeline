# SemiField Annotation Pipeline

## TODOs

1. ~~Automate autoSfM results and pipeline triggering~~
2. Create cron job for processing batches dynamically (updating a list of processed batches)
3. Make slight adjustments in data output structure to match blob container structure (cutouts in own container or within batches?)
4. Use CPU for `localize_plants`
5. Get unique bbox for cutouts in `segment_vegetation`
6. Filter cutouts to eliminate very small results in `segment_vegetation`
7. Improve documentation throughout

<br>

## Dependencies


After cloning this repo, setup your environment using conda and the provided dependency file. [Conda install instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

Run using:
```
conda env create --file=environment.yml
```

<br>

## Data Directory Structure Overview

More information about data directory structure can be found in the main data folder [blob_container](blob_container/README.md)

<br>

<p align="center">
<img src="Assets/overview.png" width=50%/>
</p>
<br>

## Run the pipeline

Settings in `conf/config.yaml` can be modified straight from the command line. 

```
python SEMIF.py general.task=segment_vegetation
                general.vi=exg
                general.clear_border=True
```

<br>
<br>

# Global bbox mapping with AutoSfM

More info can be found [here](bbox/README.md)

