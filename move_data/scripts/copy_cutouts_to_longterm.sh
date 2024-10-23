#!/usr/bin/env bash

LOGFILE="./pipeline.log"

# Copies local cutouts and associated data to longterm storage.

BATCHES=$1 # txt file of cutout batches that you want to upload to semifield-cutouts

# Paths for developed images from the previous script
PRIMARY_DEV_IMAGES="/mnt/research-projects/s/screberg/longterm_images/semifield-developed-images"
SECONDARY_DEV_IMAGES="/mnt/research-projects/s/screberg/GROW_DATA/semifield-developed-images"
TERTIARY_DEV_IMAGES="/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images"

# Paths for cutouts
PRIMARY_DST="/mnt/research-projects/s/screberg/longterm_images/semifield-cutouts"
SECONDARY_DST="/mnt/research-projects/s/screberg/GROW_DATA/semifield-cutouts"
SRCPARENT="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts"

for line in $(cat $BATCHES); do
    echo "Processing cutouts $line for longterm storage" >> $LOGFILE

    SRC=$SRCPARENT/$line
    CSVFILE="$SRC/$line.csv"

    # Check if the CSV file exists for the batch
    if [ ! -f "$CSVFILE" ]; then
        echo "CSV for $line not present. Skipping batch $line." >> $LOGFILE
        continue
    fi
    
    # Determine the destination based on where "meta_masks" is present in the developed images
    DEV_IMAGES_BATCH_PRIMARY=$PRIMARY_DEV_IMAGES/$line/meta_masks
    DEV_IMAGES_BATCH_SECONDARY=$SECONDARY_DEV_IMAGES/$line/meta_masks
    DEV_IMAGES_BATCH_TERTIARY=$TERTIARY_DEV_IMAGES/$line/meta_masks

    if [ -d "$DEV_IMAGES_BATCH_PRIMARY" ]; then
        DST_BATCHDIR=$PRIMARY_DST
        echo "Primary destination contains developed images for batch $line. Copying cutouts to primary." >> $LOGFILE
    elif [ -d "$DEV_IMAGES_BATCH_SECONDARY" ]; then
        DST_BATCHDIR=$SECONDARY_DST
        echo "Secondary destination contains developed images for batch $line. Copying cutouts to secondary." >> $LOGFILE
    
    elif [ -d "$DEV_IMAGES_BATCH_TERTIARY" ]; then
        DST_BATCHDIR=$DEV_IMAGES_BATCH_TERTIARY
        echo "Tertiary destination contains developed images for batch $line. Copying cutouts to tertiary." >> $LOGFILE
    else
        echo "Neither primary nor secondary nor tertiary destination contains developed images for batch $line. Skipping." >> $LOGFILE
        continue
    fi

    # Copy the cutout data to the determined destination
    echo "Copying cutouts for batch $line to $DST_BATCHDIR" >> $LOGFILE
    cp -r $SRC $DST_BATCHDIR

    echo "Done copying cutouts $line to longterm storage: $DST_BATCHDIR" >> $LOGFILE
    echo

done
