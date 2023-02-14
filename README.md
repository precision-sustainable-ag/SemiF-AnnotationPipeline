
# SemiField Annotation Pipeline

## Run the pipeline

1. [Setup keys](./keys/README.md)

<br>

2. Manually list batches

    Create a file (`.batchlogs/unprocessed.txt`) and list the batches you want processed. Batches must be in the same "season". List the batches with one batch per line, for example:

    **_NOTE:_**  The listed batches must be in the `semifield-developed-images` blob storage container.

    ```txt
    MD_2022-06-22
    MD_2022-06-24
    NC_2022-06-27
    TX_2022-06-28
    ```

    :warning: **Batches from different seasons cannot be processed together.** 

    **_NOTE:_**  If only listing one batch, it must be followed by an empty line.

<br>

3. Define season in [config.yaml](./conf/config.yaml#L25)

    Choose which "season" or type of crop to process. The season must correctly correspond with the listed batches in `.batchlogs/unprocessed.txt`. All batches must be of a single season.

    The [`general.season`](./conf/config.yaml#L25) must be written exactly as one of currently available seasons, defined below:

    1. `summer_weeds_2022`
    2. `cool_season_covers_2022_2023`
 
 <br>

4. Other config settings
    
| process | setting | description | default
| :---: | :---: | :---: | :---: |
AutoSfM | [downscale.factor](./conf/asfm/asfm.yaml#L38) | Input image downscaling factor (0-1) | 0.5
AutoSfM | [align_photos.downscale](./conf/asfm/asfm.yaml#L41) |  Image alignment accuracy, lower is better | 1
AutoSfM | [depth_map.downscale](./conf/asfm/asfm.yaml#L46) |  Depth map quality, lower is better | 4

<br>

5. Run the pipeline

    ```shell
    ./execute.sh
    ```
<br>

6. Monitor Logs in `.batchlogs`

<br>

-----

<br>

# Release Notes:

## Cool season cover crops 2022/23

1. Removed `dap` (days after planting) metadata information in `image` and `cutout` metadata
2. added `season` metadata field in `cutouts`
3. added `cropout_mean`, `cutout_mean`, `cropout_std`, and `cropout_std` metadata for individual cutout images


<br>

-----

<br>

## Confluence Links

This repo contains the code necessary to automatically annotate batches of incoming images from 3 location around the US and was designed specifically for the Semi-field image acquisition protocols developed by the [PSA network](https://www.precisionsustainableag.org/).

Details about the project and repo documentation can be found on the SemiField private Confluence page. Direct links to various sections can be found below.

- [Overview](https://precision-sustainable-ag.atlassian.net/l/cp/KvWLivGW)
  
- [Pipeline Description](https://precision-sustainable-ag.atlassian.net/wiki/spaces/SAP/pages/151945228/Pipeline+Description)
    
- [Setup](https://precision-sustainable-ag.atlassian.net/wiki/spaces/SAP/pages/152077232/Setup)
  
- [Pipeline Execution](https://precision-sustainable-ag.atlassian.net/wiki/spaces/SAP/pages/153911297/Pipeline+Execution)
  
- [Data Products and Metadata](https://precision-sustainable-ag.atlassian.net/wiki/spaces/SAP/pages/159711242/Data+Products+and+Metadata)