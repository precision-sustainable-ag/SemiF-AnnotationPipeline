#!/usr/bin/env bash

# Copies local cutouts and associated data to longterm storage.


BATCHES=$1 # txt file of cutout batches that you want to upload to semifield-cutouts

SRCPARENT="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts"
DST="/mnt/research-projects/s/screberg/longterm_images/semifield-cutouts"

for line in `cat $BATCHES`; do
    
    SRC=$SRCPARENT/$line
    CSVFILE="$SRCPARENT/$line/$line.csv"

    if [ ! -f "$CSVFILE" ]; then
        echo "CSV for $line not present. Exiting"
        exit 1
    fi
    echo "Copying $line to longterm storage"
    cp -r $SRC $DST
    
done



