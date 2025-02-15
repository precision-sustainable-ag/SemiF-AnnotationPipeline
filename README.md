# Current SemiF-AnnotationPipeline Execution and Validation Process


## Table of Contents
1. [Prior Seasonal Setup](#prior-seasonal-setup)
2. [Environment Setup](#environment-setup)
3. [Running ASFM](#running-asfm)
4. [Running Bbox Adjustments, Non-Target Weed Classification, and Remapping Labels](#running-bbox-adjustments-non-target-weed-classification-and-remapping-labels)
   - [Handling Pot Movement and Marker Location Issues](#handling-pot-movement-and-marker-location-issues)
5. [Running Assign Species, Convert Metadata, and Segment Vegetation](#running-assign-species-convert-metadata-and-segment-vegetation)
6. [Running Validation](#running-validation)
7. [Handling Validation Failures](#handling-validation-failures)

## Prior Seasonal Setup
Each season requires a preparatory setup before running the pipeline for individual batches. This includes:

- **Measuring Marker Locations**: Markers must be accurately positioned and recorded. In current bbot versions, this is done once (not seasonally) for each location and markers are now permanently fixed to the potting area. Past season did not have fixed markers except for in Texas. 
- **Creating a Shapefile for Potting Locations**: A single orthomosaic containing all potting groups must be successfully reconstructed. The known marker positions are required for this step.
- **GIS Mapping in QGIS**:
  - The orthomosaic is imported into GIS software (QGIS).
  - Shapefile polygons are created to delineate each potting group.
  - This shapefile is used to map individual bounding boxes, with each corner reprojected into global coordinates to assign species labels correctly.
  
- **Maintaining Fixed Potting Groups and Markers**:
  - Once this mapping is complete, potting groups and markers **must not move**.
  - Any unnotified movement of potting groups or markers will result in extensive debugging efforts due to misalignment issues, as will be discussed later.

Once this shapefile is created, this, and future batches within this season can be processed to completion.

## Environment Setup
To set up the necessary dependencies for running the AutoSfM pipeline:
```bash
conda env create -f ./environment.yml
conda activate annot
pip install ./autoSfM/package/dependencies/Metashape-2.1.2-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl
```
**Note**: the Metashape package has to be installed manually. Run the above `pip install` command, and all future commands, from the repo root.

## Running ASFM
To execute the pipeline for a given batch:
```bash
python PIPELINE.py general.batch_id=MD_2023-03-20 \
    general.season=cool_season_cover_2022_2023_MD_pos_2 \
    pipeline.download_data=True \
    pipeline.autosfm_pipeline=True \
    pipeline.merge_overlapping_bboxes=False \
    pipeline.classify_nontarget_weeds=False \
    pipeline.localize_plants=False \
    pipeline.remap_labels=False \
    pipeline.assign_species=False \
    pipeline.convert=False \
    pipeline.segment_vegetation=False \
    asfm.downscale.factor=0.5 \
    asfm.align_photos.downscale=4 \
    asfm.depth_map.downscale=4
```

### Manual Inspection and Troubleshooting
After the AutoSfM pipeline completes, results must be manually inspected. If issues arise, they may require re-running the pipeline with adjusted parameters.

#### Common Issues:
- **Bad Alignment or Failed Alignment**: Caused by insufficient image overlap.
- **Inaccurate Scaling**: Due to misplaced markers.
- **Overlapping Reconstructed Sections**: Arises from marker changes (added, removed, or moved).

#### Troubleshooting Steps:
- If alignment excludes some images, decrease the downscale setting for `align_photos`.
- If scaling is inaccurate but not widespread, minimal debugging is performed.
- If reconstructed sections overlap, inspect orthomosaic results in QGIS alongside marker locations. If necessary, estimate marker positions, update the marker sheet, and re-run AutoSfM.

## Running Bbox Adjustments, Non-Target Weed Classification, and Remapping Labels
If the reconstructed batch is satisfactory, proceed with:
```bash
python PIPELINE.py general.batch_id=MD_2023-03-20 \
    general.season=cool_season_cover_2022_2023_MD_pos_2 \
    pipeline.download_data=False \
    pipeline.autosfm_pipeline=False \
    pipeline.merge_overlapping_bboxes=True \
    pipeline.classify_nontarget_weeds=True \
    pipeline.localize_plants=True \
    pipeline.remap_labels=True \
    pipeline.assign_species=False \
    pipeline.convert=False \
    pipeline.segment_vegetation=False
```

### Remap Labels Step Considerations
- This step can sometimes hang. If it does, restart the process.
- Issues may stem from poor reconstruction, causing incorrect reprojection of pixel coordinates.
- Debugging involves:
  - Checking log files for problematic images.
  - Inspecting the AutoSfM project file for reconstruction issues.
  - Verifying orthomosaic results in GIS software.
- If remap_labels fails consistently, running it separately from other steps usually resolves the issue.

### Handling Pot Movement and Marker Location Issues
- If issues arise because of reconstruction problems related to marker locations or pot movement (found from investigating the orthomosaics overlaid with marker locations), and if it's found that pot locations did move, as is often the case in early-season years, then a second shapefile will need to be created to account for the new species group locations.
- This is why this batch has `general.season=cool_season_cover_2022_2023_MD_pos_2`, as it, along with other batches, was found to have different potting positions than prior batches in the same `cool_season_cover_2022_2023` season.

## Running Assign Species, Convert Metadata, and Segment Vegetation
```bash
python PIPELINE.py general.batch_id=MD_2023-03-20 \
    general.season=cool_season_cover_2022_2023_MD_pos_2 \
    pipeline.download_data=False \
    pipeline.autosfm_pipeline=False \
    pipeline.merge_overlapping_bboxes=False \
    pipeline.classify_nontarget_weeds=False \
    pipeline.localize_plants=False \
    pipeline.remap_labels=False \
    pipeline.assign_species=True \
    pipeline.convert=True \
    pipeline.segment_vegetation=True
```
### Common Issues and Troubleshooting

The segment vegetation can sometimes hang. In this case, I'll kill the process and re-run this step. This usually fixes the issue.

## Running Validation

After pipeline completion, results are inspected:
- A sample of images, masks, bounding boxes, and cutouts are reviewed.
- The sample includes images with at least two species (potting group boundaries).
- X11 forwarding is required to visualize the images.
- Users validate images by pressing:
    - `a` (Pass)
    - `s` (Fail)
    - `t` (Review)
- Results are saved in a CSV file.

If validation passes, batch data is:
- Marked as `validated==True`
- Moved to long-term storage (LTS)
- Local batch data is removed

User confirmation is required at each step.


```bash
python validate/inspect_validation.py general.batch_id=MD_2023-03-20 \
    validate.sample_sz=50
```

## Handling Validation Failures
- If validation fails, investigation is required.
- Most issues stem from mislabeled species or poorly segmented vegetation.
- If a small number of images (<30) are affected, data is removed using:

```bash
python correct/remove_data.py \
    correct.start_epoch=1667495929 \
    correct.end_epoch=1667496083
```
This script removes:
1. The main developed metadata for affected images.
2. Any associated cutout data (cropout, color cutout segment, segment mask, metadata).
3. The rows associated with the image_id in the batch cutout summary CSV.
4. References to affected image IDs in overlapping cutout metadata.

This ensures bad data is completely removed, making it ideal for small-scale issues rather than extensive debugging.

The validation process would then need to be re-run for final data validation and transfer to LTS.