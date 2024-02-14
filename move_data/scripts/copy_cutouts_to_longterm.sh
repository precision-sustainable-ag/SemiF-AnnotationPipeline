#!/usr/bin/env bash

# Copies local cutouts and associated data to longterm storage.


BATCHES=$1 # txt file of cutout batches that you want to upload to semifield-cutouts

SRCPARENT="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts"
DST="/mnt/research-projects/s/screberg/GROW_DATA/semifield-cutouts"

for line in `cat $BATCHES`; do
    echo
    echo "Copying cutouts $line to longterm storage"

    
    SRC=$SRCPARENT/$line
    CSVFILE="$SRCPARENT/$line/$line.csv"

    if [ ! -f "$CSVFILE" ]; then
        echo "CSV for $line not present. Exiting"
        exit 1
    fi
    
    cp -r $SRC $DST
    echo "Done copying cutouts $line to longterm storage"
    echo
    
done



