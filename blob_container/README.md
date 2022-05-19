### The WIR is made up of five blob containers
<br>

<p align="center">
<img src="../Assets/overview.png" width=125%/>
</p>
<br>


## Upload

<details>
<summary>Expand to view layout</summary>
<p align="center">
<img src="../Assets/Uploaded.png"  width=60%/>
</p>
</details>
<br>

## Developed

<details>
<summary>Expand to view layout</summary>
<p align="center">
<img src="../Assets/Developed_blob.png"  width=75%/>
</p>
</details>
<br>

## Cutouts

<details>
<summary>Expand to view layout</summary>
<p align="center">
<img src="../Assets/Cutouts_blob.png"  width=60%/>
</p>
</details>
<br>

## Synthetic

<details>
<summary>Expand to view layout</summary>
<p align="center">
<img src="../Assets/Synthetic_blob.png"  width=75%/>
</p>
</details>
<br>


## Models

<details>
<summary>Expand to view layout</summary>
<p align="center">
<img src="../Assets/models_blob.png"  width=40%/>
</p>
</details>
<br>




Directory and path organization for data and products.

```
SemiF-AnnotationPipeline
└── data
    ├── trial
    │   ├── NC_2022-03-11
    │   │   ├── developed
    │   │   │   ├── row1_1.jpg
    │   │   │   └── ...
    │   │   ├── cutouts
    │   │   │   ├── row1_1_0.png
    │   │   │   └── ...
    │   │   ├── autosfm
    │   │   │   ├── ortho
    │   │   │   |   └── orthomosaic.tif
    │   │   │   └── reference
    │   │   │       ├── camera_reference.csv
    │   │   │       └── ...
    │   │   ├── labels
    │   │   │   ├── row1_1.json
    │   │   │   └── ...
    │   │   ├── GroundControlPoints.csv
    │   │   └── detections.csv
    │   │   
    │   ├── NC_2022-03-29
    │   └── ...
    │   
    └── synth_data
        ├── bench
        |   ├── NC_row1_1650664275512.jpg
        │   └── ...
        └── pots
            ├── NC_pot1_1650664272865.png
            └── ... 
  ```