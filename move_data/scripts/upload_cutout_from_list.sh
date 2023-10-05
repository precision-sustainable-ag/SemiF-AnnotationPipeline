#!/usr/bin/env bash

# Uploads cutout to azure blob storage (semifield-cutouts)


BATCHES=$1 # txt file of batches that you want to upload to cutouts

SAS=$(cat /home/psa_images/SemiF-AnnotationPipeline/keys/pipeline_keys.yaml | shyaml get-value SAS.cutouts.upload)

for line in `cat $BATCHES`; do

    prefix=${SAS%%?sv=*} # remove before ?sv=
    suffix=${SAS#*?sv=} # remove after   ?sv=
	src="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts/$line"
	CSVFILE="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts/$line/$line.csv"
	
	if [[ -d "$src" && -f "$CSVFILE" ]]; then
		echo
    	echo "Local source: $src is a directory."
		echo "$CSVFILE exists."
		echo "Azure destination: $SAS is a directory."
		azcopy copy "${src-}" "${SAS-}" --recursive
	else
		echo "Local source: $src is NOT a directory."

	fi
done