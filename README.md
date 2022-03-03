# SemiF-AnnotationPipeline

Collecting and labeling images of weeds is time-consuming and costly, and has severely slowed the development of datasets necessary for utilizing artificial intelligence in agriculture, particularly in weed identification, mapping and precision management applications. This project aims to reduce the burden of manually annotating images of weeds and other plants by implementing a semi-automatic annotation pipeline designed to iterate over batches of incoming images.

1. Images collected from BenchBot uploaded to shared storage
2. Classical image processing techniques are applied to segment vegetation creating a library of plant cutouts
3. Synthetic data is generated from cutouts and used to train both annotation assistant networks
4. AutoSfM (C.M. Savadikar 2022) takes in batched image uploads and returns a global corrdinate reference system(CRS) that tags individual pots for tracking and custom annotation configuration
5. Annotation assistant models
   1.  The detection assistant network is trained on a combination of synthetic data and real data. Local detection results will be mapped to a global CRS provided by AutoSfM
   2.  The segmentation assistant network is trained on a combination of synthetic data and real data from previsous batches to predict on real data to generate sematic annotations
6. Bounding box and semantic annotations are generated from synthetic and real data.

No images need to be manually annotated. Only it the first iterations will cutouts need to be manually sorted into distinct classes. This process aims to save an estimated 22 hours per 1,000 segmentations. Automatic annotation pipelines play an important role in developing robust datasets for trained AI models that can handle diverse scenes. The methodology devised here will be utilized in a weed image library pipeline currently being developed by the [Precision Sustainable Ag](https://precisionsustainableag.org/) research network.

---

## Simple Flowchart
![](Assets/semif_pipeline_v4_simplified_med.png)

---

## Detailed Flowchart
![](Assets/semif_pipeline_v4_small.png)


## Metadata

```YAML

Per site:
   - Site-ID:
   - QR-marker location:
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