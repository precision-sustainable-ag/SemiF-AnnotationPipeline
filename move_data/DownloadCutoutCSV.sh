#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail


processed=./.batchlogs/processed.txt
download=SAS_download_cutout_key.txt

while read line; do
    while read key; do

    prefix=${key%%semifield-cutouts*}
    suffix=${key#*semifield-cutouts} 
    src=${prefix-}semifield-cutouts/${line-}/${line-}.csv${suffix-}

    dest=./SemiF-AnnotationPipeline/data/semifield-cutouts/${line-}
    if [ ! -d $dest ]; then
        # mkdir -p $dest
        echo
        echo $src
        echo "..."
        echo $dest
        
        # azcopy copy ${src} ${dest}
    fi

    done <$download

done <$processed
