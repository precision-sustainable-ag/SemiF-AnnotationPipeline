
# SemiF-AnnotationPipeline

<br>

## Table of Contents  
[Dataflow](#dataflow)
[Setup](#setup)  
[Configure](#configure)  
[Run](#run)
[About](#about)

<br>

## Dataflow

![](assets/dataflow_repo.png)


## Setup

### 1. Clone this remote

```
git clone https://github.com/precision-sustainable-ag/SemiF-AnnotationPipeline.git
```

### 2. Look for NFS mount

```
df /mnt/research-projects/s/screberg/longterm_images/ -TP  | tail -n -1 | awk '{print $2}'
```
Output should say "`nfs4`"

## Configure

### Set Keys

#### SAS Download

* Only `read` privileges
* One for Cutouts and Developed-images


#### SAS Upload

#### Metashape license


### Choose Season
Manual enter one of the following:
- summer_weeds_2022
- coolseason_covercrops_2022_2023
- spring_weeds_2023
- spring_cashcrops_2023
- summer_weeds_2023

for `general.season` in [conf/config.yaml](conf/config.yaml).

### Find unprocessed batches



#### Download Batch








## Run

### 1. [AutoSfM](autoSfM/README.md)

### 2. [Move_data](move_data/README.md)

### 3. [Segment](segment/README.md)

### 4. [Inspect](inspect/README.md)

## About

### [AutoSfM](autoSfM/README.md)

### [Move_data](move_data/README.md)

### [Segment](segment/README.md)

### [Inspect](inspect/README.md)