#!/usr/bin/env bash

LOGFILE="./pipeline.log"

# Copies local batch data to longterm storage.

BATCHES=$1 # txt file of batches that you want to upload to cutouts

PRIMARY_DST="/mnt/research-projects/s/screberg/longterm_images/semifield-developed-images"
SECONDARY_DST="/mnt/research-projects/s/screberg/GROW_DATA/semifield-developed-images"
TERTIARY_DST="/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images"
SRCPARENT="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images"

for line in $(cat $BATCHES); do
    echo "Starting batch $line to longterm storage" >> $LOGFILE

    SRC=$SRCPARENT/$line
    DST_BATCHDIR_PRIMARY=$PRIMARY_DST/$line
    DST_BATCHDIR_SECONDARY=$SECONDARY_DST/$line
    DST_BATCHDIR_TERTIARY=$TERTIARY_DST/$line

    # Determine destination based on which one already contains images or none
    if [ -d "$DST_BATCHDIR_PRIMARY/images" ]; then
        DST_BATCHDIR=$DST_BATCHDIR_PRIMARY
        echo "Primary destination already contains images. Skipping image copy to primary." >> $LOGFILE
    elif [ -d "$DST_BATCHDIR_SECONDARY/images" ]; then
        DST_BATCHDIR=$DST_BATCHDIR_SECONDARY
        echo "Secondary destination already contains images. Skipping image copy to secondary." >> $LOGFILE
    
    elif [ -d "$DST_BATCHDIR_TERTIARY/images" ]; then
        DST_BATCHDIR=$DST_BATCHDIR_TERTIARY
        echo "Tertiary destination already contains images. Skipping image copy to tertiary." >> $LOGFILE
    else
        # If no images in primary or secondary, copy to primary
        DST_BATCHDIR=$DST_BATCHDIR_PRIMARY
        echo "Neither primary nor secondary destination contains images. Setting destination to primary." >> $LOGFILE
    fi

    # Create destination directory if it does not exist
    if [ ! -d "$DST_BATCHDIR" ]; then
        echo "Making destination directory $DST_BATCHDIR" >> $LOGFILE
        mkdir -p $DST_BATCHDIR
    fi
    
    # Define paths for each subfolder
    DSTimages="$DST_BATCHDIR/images"
    images="$SRC/images"
    asfm="$SRC/autosfm/reference"
    meta_masks="$SRC/meta_masks"
    metadata="$SRC/metadata"

    # Copy images only if not already present in destination
    if [ -d "$DSTimages" ]; then
        echo "Image directory exists in $DST_BATCHDIR. Skipping image copy." >> $LOGFILE
    else
        echo "Copying images to $DSTimages" >> $LOGFILE
        cp -r $images $DSTimages
    fi

    # Copy additional directories regardless
    echo "Copying asfm reference data to $DST_BATCHDIR" >> $LOGFILE
    cp -r $asfm $DST_BATCHDIR

    echo "Copying meta_masks to $DST_BATCHDIR" >> $LOGFILE
    cp -r $meta_masks $DST_BATCHDIR

    echo "Copying metadata to $DST_BATCHDIR" >> $LOGFILE
    cp -r $metadata $DST_BATCHDIR

    echo "Done copying to longterm storage for batch $line" >> $LOGFILE
    echo
done
