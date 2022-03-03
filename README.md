# SemiF-AnnotationPipeline

Collecting and labeling images of weeds is time-consuming and costly, and has severely slowed the development of datasets necessary for utilizing artificial intelligence in agriculture, particularly in weed identification, mapping and precision management applications. This project aims to reduce the burden of manually annotating images of weeds and other plants by implementing a semi-automatic annotation pipeline designed to iterate over batches of incoming images.

1. Images collected from BenchBot uploaded to shared storage
2. Classical image processing techniques will be applied to segment vegetation creating a library of plant cutouts
3. Synthetic data will be generated from cutouts and will be used to train both annotation assistant networks
4. Annotation assistant models
   1.  Detection assistant  models
5. Using detection results and simple image processing techniques to extract and classify vegetation.

No images need to be manually annotated, only cutouts need to be manually sorted into distinct classes, saving an estimated 22 hours per 1,000 cut-outs. Automatic annotation pipelines play an important role in developing robust datasets for trained AI models that can handle diverse scenes. The methodology devised here will be utilized in a weed image library pipeline currently being developed.

## Simple Flowchart
![](Assets/semif_pipeline_v4_simplified_med.png)

## Detailed Flowchart
![](Assets/semif_pipeline_v4_small.png)


