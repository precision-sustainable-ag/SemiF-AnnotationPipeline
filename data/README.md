# WIR is made up of five blob containers

The WIR is made up of four main datasets and a temporary upload data container.

Batch = Images collected from an individual sites on a single day. (e.g. NC_2022-03-29)

Batches make up the first data heirarchy for 3 of the dataset.

While blob container don't technically use directory/subdirectory convention, we organize data within each container using psuedo-directories (`/`) to organize data using a heirarchy of subdirectories.

<br>

<p align="center">
<img src="../Assets/overview.png" width=100%/>
</p>

<br>



# Upload
The first location for images after being uploaded by Benchbot operators. Images are manually uploaded by dragging and dropping "batch" files (organized and named by the bbot GUI) into this blob container. The [`upload`](uploaded_example) container organizes images into a single heirarchy defined by batches. Each [`batch`](uploaded_example/NC_2022-04-12/) contains:
1. Raw images
   - images are automatically named by the GUI before being uploaded
   - [site_id]\_[row]\_[stop]_[unix time in millieseconds].ARW
2. Ground_contol_points.csv
   - a single file with location of markers in meters 
   -  | marker ID | X      | Y      | Z      |
      | --------- | ------ | ------ | ------ |
      | target 1	| 0	   | 0	   | -0.213 |
      | target 2	| 1.05	| 0	   | -0.213 |
      | target 3	| 2.2	   | 0	   | -0.213 |
3. Pot Species map
   - TBD

```
uploaded
   └── NC_2022-04-12
      ├── GroundControlPoints.csv
      ├── NC_2022-04-12_1_1_1649779201.024.ARW
      ├── NC_2022-04-12_1_2_1649779202.024.ARW
      └── ...
```
<details>
<summary>Figure</summary>
<p align="center">
<img src="../Assets/Uploaded.png"  width=50%/>
</p>
</details>
<br>
<br>

# Developed

After preprocessing, batches are moved to "Developed". Each batch contains four subdirectories. As images move through the processing pipeline, these subdirectories are populated with metadata and label information. Subdirectories, their contents, and origin are as follows:
1. [autosfm/](../blob_container/developed_example/NC_2022-03-29/autosfm/)
   - [`autosfm.py`](../auto_sfm.py) **outputs** four csv files that serve as **inputs** for [`remap_labels.py`](../remap_labels.py).
     - fov.csv, gcp_reference.csv, error_statistics.csv, and camera_reference.csv
   - detections.csv 
     - obeject detection results **output** from [`localize_plants`](../localize_plants.py)
2. [images/](../blob_container/developed_example/NC_2022-03-29/images/)
   - \[site_id]\_[row]\_[site]\_[unix time in millieseconds].jpg
   - Ex. NC_1_1_1647016472000.0.jpg
3. [metadata/](../blob_container/developed_example/NC_2022-03-29/metadata/)
4. [batch model (.json)](../blob_container/developed_example/NC_2022-03-29/NC_2022-03-29.json)
   - json file that records batch upload information including:
     - list of images in batch
     - pseudo directory storage location
     - and various metadata


```
developed
   └── NC_2022-03-29
         ├── autosfm
         │   ├── downscaled_photos
         │   │    ├── NC_2_1_1648579721000.0.jpg
         │   │    └── ...
         │   ├── ortho
         │   │    └── ortho.tif
         │   ├── camera_reference.csv
         │   ├── detections.csv
         │   ├── error_statistics.csv
         │   ├── fov.csv
         │   ├── gcp_reference.csv
         │   └── run4measurements.txt
         ├── images
         │   ├── NC_2_1_1648579721000.0.jpg
         │   └── ...
         ├── metadata
         │   ├── NC_2_1_1648579721000.0.json
         │   └── ...
         └── NC_2022-03-29.json

```
<details>
<summary>Figure</summary>
<p align="center">
<img src="../Assets/Developed_blob.png"  width=75%/>
</p>
</details>
<br>
<br>

# Cutouts

Batches of vegetation cutouts are created from `segment_vegetation.py`. Each batch subdirectory contains cutout iamges and metadata.
Cutotu naming convention:
- [site_id]\_[row]\_[stop]\_[unix time in milliseconds]\_[cutout number].png
- [site_id]\_[row]\_[stop]\_[unix time in milliseconds]\_[cutout number].json

Each [cutout model object (.json)] contains:
- pseudo directory location information
- metadata
- and cutout property information
- Ex. [NC_2_1_1648579721000.0_18.json](cutouts_example/NC_2022-03-29/NC_2_1_1648579721000.0_18.json)


```
cutouts
   └── NC_2022-03-29
      ├── NC_2_1_1648579721000.0_18.json
      ├── NC_2_1_1648579721000.0_18.png
      └── ...
```
<details>
<summary>Figure</summary>
<p align="center">
<img src="../Assets/Cutouts_blob.png"  width=60%/>
</p>
</details>
<br>
<br>


# Synthetic

Synthetic data serves as storage location for most inputs and all outputs of [`synthesize.py`](../synthesize.py). Cutouts are the only input of `synthesize.py` not stored in this container. They are sourced from the [`cutouts`](cutouts_example) container.

The [`synthetic`](synthetic_example) container is made up of:

1. [backgrounds](synthetic_example/backgrounds) (**inputs**)
   - "empty" images of only weed fabric. No pots or plants
   - image and json model
2. [pots](synthetic_example/pots) (**inputs**)
   - "empty" pots without vegetation
   - image and json model
3. [images](synthetic_example/images) (**outputs**)
   - images only
4. [masks](synthetic_example/masks) (**outputs**)
   - images only
5. [metadata](synthetic_example/metadata) (**outputs**)
   - json model only

```
synthetic
   ├── backgrounds
   │   ├── 862f315672e94d30817809da4d109c48.png
   │   ├── 862f315672e94d30817809da4d109c48.json
   │   └── ...
   ├── pots
   │   ├── d8453b13ab7b424fac446818b491bdf2.png
   │   ├── d8453b13ab7b424fac446818b491bdf2.json
   │   └── ...
   ├── images
   │   ├── 059fd6a9feb24fb1a96c320d0bd2086f.png
   │   └── ...
   ├── masks
   │   ├── 059fd6a9feb24fb1a96c320d0bd2086f.json
   │   └── ...
   └── metadata
      ├── 059fd6a9feb24fb1a96c320d0bd2086f.png
      └── ...
```

<details>
<summary>Figure</summary>
<p align="center">
<img src="../Assets/Synthetic_blob.png"  width=75%/>
</p>
</details>

<br>
<br>

# Models

<br>
<details>
<summary>Figure</summary>
<p align="center">
<img src="../Assets/models_blob.png"  width=40%/>
</p>
</details>

<br>






<details>
<summary>Figure</summary>
Directory and path organization for data and products.

```








models_example
   └── plant_detector
         └── best.pt


  ```
</p>
</details>