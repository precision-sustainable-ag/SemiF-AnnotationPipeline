#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# localsrc=$1
# data=$2

 filename=SAS_upload_key_cutout.txt
for dir in ./SemiF-AnnotationPipeline/data/semifield-cutouts/*
do
    while read line; do
    # Appends data to key destination
        
        echo "Source:"
        echo $dir
        echo
        echo "Destination:"
        echo $line
        echo
        azcopy copy ${dir} ${line} --recursive 


        
        done <$filename
done