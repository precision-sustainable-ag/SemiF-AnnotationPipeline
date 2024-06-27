#!/usr/bin/env bash

LOGFILE="./pipeline.log"

# Uploads cutout to azure blob storage (semifield-cutouts)

BATCHES=$1 # txt file of batches that you want to upload to cutouts

SAS=$(cat /home/psa_images/SemiF-AnnotationPipeline/keys/pipeline_keys.yaml | shyaml get-value SAS.cutouts.upload)

for line in `cat $BATCHES`; do
	echo
	echo "Uploading cutouts $line to Azure semifield-cutouts" >> $LOGFILE
    prefix=${SAS%%?sv=*} # remove before ?sv=
    suffix=${SAS#*?sv=} # remove after   ?sv=
	src="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts/$line"
	CSVFILE="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-cutouts/$line/$line.csv"
	
	if [[ -d "$src" && -f "$CSVFILE" ]]; then
		echo "Local source is a directory and cutouts csv file exists." >> $LOGFILE
		azcopy copy "${src-}" "${SAS-}" --recursive >> $LOGFILE 2>&1
		echo "Done uploading cutouts $line to Azure blob container semifield-cutouts" >> $LOGFILE
	else
		echo "Local source is NOT a directory." >> $LOGFILE
		echo "DID NOT MOVE TO AZURE" >> $LOGFILE
	fi
	echo
done