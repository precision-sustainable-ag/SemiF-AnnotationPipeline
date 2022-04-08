

Collecting and labeling images of weeds is time-consuming and costly, and has severely slowed the development of datasets necessary for utilizing artificial intelligence in agriculture, particularly in weed identification, mapping and precision management applications. This project aims to reduce the burden of manually annotating images of weeds and other plants by implementing a semi-automatic annotation pipeline designed to iteratively update annotation assistant models over batches of incoming images.

# SemiField Annotation Pipeline

## Config and Run
Uses Hydra 
```
pip install hydra-core
```

Set config file variables in `conf/config.yaml`. See below for task variable specifics

Run by using
```
python SEMIF.py
```

## Data Structure
<details open>

Ideally Metadata and images are passed through the pipeline together so that they can be easily accessed to customize and thus improve processing.
For example, an image as a dataclass can point to physical image location, but also store other data attributes like bounding box/species information. When segmenting vegetation, one algorithm may be preferable over the other depending on the detected species, growth stage, etc.

```Python
@dataclass
class ImageData:
    """ Data and metadata for individual images """
    image_id: str 
    image_path: str
    upload_id: str # Points to batch metadata that can be used to customize image processing
    bboxes: List[BBox]
    camera_data: CameraData = field(default=None)
    width: int = field(init=False, default=-1)
    height: int = field(init=False, default=-1)
```

![](Assets/data_struct_v1.png)

TODO Provide more details


</details>
<br>

## Localize Plants
<details>
<summary>A trained detection model is use to localize and segment vegetation from incoming images</summary>

Input:
```YAML
# conf/confgi.yaml
general:
   task: localize_plants
   model_path: ./path/to/detection/model
   csv_savepath: ./path/to/save/csv/detection file
   imagedir: ./path/to/images
```

Outputs: 
* Bounding box csv file containing

</details>
<br>

## Segment Vegetation
<details>
<summary>Localize and segment plants from within bounding box detection results. Apply classical image processing techniques - VIs, thresholding, and unsupervised clustering - to classify pixels.</summary>

```YAML
general:
   task: segment_vegetation
   datadir: path to root data directory that contains benchbot images
   savedir: path to save "masks" and "cutouts"
   num_class: number of species classes
   vi_name: vegetation index
   class_algo: classification algorithm
```

Output:
1. pixel-wise masks of vegetation
2. vegetation cutouts

</details>
<br>

## Synthesize Data
<details>
<summary>Create a dataset of synthetic image using vegetation cutouts.</summary>

Inputs: 
* Vegetation segments
  
Outputs:
  1. synthetic images
  2. bounding box labels (by species)
  3. pixel-level mask with species labels

</details>
<br>

## Train Plant Detector
<details>
<summary>Train plant detector</summary>

* Use synthetic data to train detection model

</details>
<br>

---
<br>

## Simple Flowchart
<details>
<summary>Simple Flowchart</summary>
<p align="center">
<img src="Assets/semif_pipeline_v4_simplified_small.png" width="750">
</p>
</details>
<br>

---

<br>

## Metadata
<details>
<summary>Metadata</summary>

Listed linearly as data is passed through the pipeline. Alternatively, heirchical representation may be used.
```YAML

Site:
   - Site-ID:
   - GCP-marker location:
   - Species location map:
Per collection:
   - Collection ID:
   - Upload ID:
   - Weather data:
   - Color calibration:
Per image:
   - Image-ID:
   - Focal length:
Per auto-SfM processing:
   - Processing ID:
   - Image location and orientation in consistent global coordinate system:
Per weed recognition processing:
   - Processing ID:
   - Whether it is automated or manually annotated BBs:
   - If model-based save the computer vision model ID as well:
   - Image-domain bounding box ID in local coordinate system:
After Bounding box transform:
   - Bounding boxes in global coordinate system:
   - Non-maximum suppressed bounding boxes to remove duplicate bounding boxes:
   - Plant species:
Per Bounding box annotation:
   - Global bounding box-ID in global coordinate system:
   - Corresponding image and corresponding local coordinates of bounding box:
   - Plant species:
   - Plant mask:
   - Plant ID:
Per unique plant:
   - Plant ID:
   - List of all images:
   - List of all bounding box image crops:
   - List of all segmentations:
   - Plant species:
   - Seed date:
   - Emergence date:
Per segmentation:
   - Method of segmentation:
   - Path to segmentation file:
```
</details>
<br>

---
<br>

## Detailed Flowchart
<details>
<summary>Detailed Flowchart</summary>
TODO update this figure
![](Assets/semif_pipeline_v4_small.png)
</details>

# SemiF-AnnotationPipeline


# Bounding Box Utilities

The directory ```bbox``` contains the utilities to map the bounding box coordinates (manual annotations/detections from a model) to the global coordinate system. This coordinate system is defined by the markers which are placed on the BenchBot. This is done in order to find out overlappig objects and deduplicate them.

The local coordinates are obtained through an XML file (manual annotations for now). Following attributes from the autoSfM are required for translating from the local coordinates to global coordinates:
- Camera Field of View (fov)
- Camera X, Y, Z coordinates for each image (camera_location)
- Pixel Height (in terms of global coordinate system units, which is meters)
- Pixel Width (in terms of global coordinate system units, which is meters)
- Yaw angle of the camera (in degrees)
- Pitch angle of the camera (in degrees)
- Roll angle of the camera (in degrees)
- Focal Length of the camera (in pixels)

These fields can be found in the ```camera CSVs``` of the autoSfM outputs.

## Data structures
The file ```bbox/bbox_utils.py``` contains several data classes used to store the bounding box and image information.
1. ```BoxCoordinates```: A utility structure for storing the top left, top right, bottom_left and bottom right X ad Y coordinates.
1. ```BBox```: Class which stores the bounding box information, which contains local and global coordinates, object class, a unique id assigned to the box, and the image ID for which the bounding box is associated. Other fields, which are not supplied by the user but are derived from the other fields are local centroid, global centrid and the ```is_primary``` flag. This flag tracks whether the bounding box is the ideal box or not (the box which is closest to the camera location).
2. ```Image```: A class containing all the metadata for each image processed by the autoSfM pipeline.

## Transformations
1. ```BBoxMapper```: Maps the bounding boxes from the local (image) coordinate system to the global coordinate system.
2. ```BBBoxFilter```: Finds the ideal bounding box based on the overlap between boxes in the global coordinate system.

## Connectors
1. ```SfMComponents```: An interface to read the CSVs from the autoSfM pipeline. This also converts the ```camera_reference.csv``` and ```fov_reference.csv``` into a common logical camera_reference DataFrame.
2. ```BBoxComponents```: An interface to convert bounding box coordinates and image metadata from the autoSfM CSVs and annotation files to the data structures defined above. BBoxComponents takes a ```reader``` function as an argument, which is responsible for reading the annotation from the annotation files (so that annotations can flexibly come from XML, JSON, etc.). The reader function must return the bounding boxes of the objects in the image in the following format:
```python
def reader(*args, **kwargs):

    # ... Do any file io

    image_list = [
        {"id": "image_id1", "path": "path/to/image1"}, 
        {"id": "image_id2", "path": "path/to/image2"},
        ...
    ]

    bounding_boxes = {
        "image_id1": [
            {
                "id": "bbox1_id",
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right,
                "cls": "bbox1_class"
            },
            {
                "id": "bbox2_id",
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right,
                "cls": "bbox2_class"
            },
            ...
        ],
        "image_id2": [
            {
                "id": "bbox1_id",
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right,
                "cls": "bbox1_class"
            },
            {
                "id": "bbox2_id",
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right,
                "cls": "bbox2_class"
            },
            ...
        ]
        ...
    }

    return (image_list, bounding_boxes)

```
