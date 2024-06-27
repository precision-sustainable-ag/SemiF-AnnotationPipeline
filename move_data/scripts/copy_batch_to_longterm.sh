#!/usr/bin/env bash

LOGFILE="./pipeline.log"

# Copies local batch data to longterm storage.

BATCHES=$1 # txt file of batches that you want to upload to cutouts

DST="/mnt/research-projects/s/screberg/GROW_DATA/semifield-developed-images"
SRCPARENT="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images"

for line in `cat $BATCHES`; do
    echo
    echo "Starting batch $line to longterm storage" >> $LOGFILE

    DST_BATCHDIR=$DST/$line
    SRC=$SRCPARENT/$line
    echo "Copying to destination: $DST_BATCHDIR" >> $LOGFILE
    if [ ! -d "$DST_BATCHDIR" ]; then
        echo "Making destination directory" >> $LOGFILE
        mkdir $DST_BATCHDIR
    fi
    
    DSTimages="$DST_BATCHDIR/images"
    images="$SRC/images"
    asfm="$SRC/autosfm/reference"
	meta_masks="$SRC/meta_masks"
	metadata="$SRC/metadata"
    jsonmetadata="$SRC/$line.json"

    if [ -d "$DSTimages" ]; then
        echo "Image directory exists. Skipping image copy." >> $LOGFILE
    else
        echo "Copying images because they do not exist in longterm storage" >> $LOGFILE
        cp -r $images $DSTimages
    fi

    echo "Copying asfm reference data" >> $LOGFILE
    cp -r $asfm $DST_BATCHDIR
    echo "Copying meta_masks" >> $LOGFILE
    cp -r $meta_masks $DST_BATCHDIR
    echo "Copying metadata" >> $LOGFILE
    cp -r $metadata $DST_BATCHDIR
    echo "Copying .json file" >> $LOGFILE
    cp $jsonmetadata $DST_BATCHDIR
    echo "Done copying to longterm storage for batch $line" >> $LOGFILE
    echo

done