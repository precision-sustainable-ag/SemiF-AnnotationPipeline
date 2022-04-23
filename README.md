
# SemiField Annotation Pipeline

## Config

Set config file variables in `conf/config.yaml`. See below for task variable specifics.

s

## Data Directory Structure
<details open>
<summary>Directory and path organization for data and products.</summary>

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
    │   │   │   ├── reference
    │   │   │   │   ├── camera_reference.csv
    │   │   │   │   └── ...
    │   │   │   └── metadata
    │   │   │       ├── row1_1.json
    │   │   │       └── ...
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

</details>

## Data Structure

Data is moved through the pipeline and directly to the mongodb using  `@dataclasses` located in `datasets.py`.

For examples:
``` Python
@dataclass
class BatchMetadata:
    upload_dir: str
    site_id: str
    upload_datetime: str
    batch_id: uuid 
    image_list: List 
    schema_version: str = "v1"
```
<br>

## Run the pipeline

`SEMIF.py` runs the pipeline based on `conf/config.yaml` saving results in the same `imagedir:` provided in the config file. 

Settings in `conf/config.yaml` can be modified straight from the command line. 

```
python SEMIF.py general.task=segment_vegetation
                general.vi=exg
                general.clear_border=True
```

<br>





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

## BBox Data structures
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