
# SemiField Annotation Pipeline

## Run the pipeline

1. [Setup keys](./keys/README.md)

2. Manually list batches  

    Create a file (`.batchlogs/unprocessed.txt`) and list the batches you want processed. Batches must be in the same "season". For example:

    ```txt
    MD_2022-06-22
    MD_2022-06-24
    NC_2022-06-27
    TX_2022-06-28
    ```
    

    :warning: **Batches from various seasons cannot be processed together.** :

3. Define season in [config.yaml](./conf/config.yaml#L25)  

    Choose which "season" or type of crop to process. 


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