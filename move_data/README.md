# MoveData

Either downloads or uploads data from blob container.

Download depends on presence of subfolders in the main folder:
- `images`
- `masks`
- `plant-detections`
- presence of `[batch_id]` in `/.batchlogs/unprocessed.txt`

Upload depends on presence of:
- `metadata`
- `meta_masks`
- `[batch_id].json`
- presence of `[batch_id]`  in  `/.batchlogs/processed.txt`