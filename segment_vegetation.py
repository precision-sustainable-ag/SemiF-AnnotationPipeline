import yaml

from datasets import ImageData, get_batchloader

# Load ImageData
metadata_sheet = "data/semif-trial/semif-trial/metadata_sheet.yaml"
meta = yaml.load(open(metadata_sheet, "rb"), Loader=get_batchloader())

# Segment vegetation

# Create Cutout

# Create PlantCutout and append Cutout

# Move to database
