#!/usr/bin/env bash

# Uploads local batch to azure blob storage (semifield-developed-images)
# autosfm, meta_masks, metadata, and *.json

BATCHES=$1
SAS=$(cat /home/psa_images/SemiF-AnnotationPipeline/keys/pipeline_keys.yaml | shyaml get-value SAS.developed.upload)

for line in `cat $BATCHES`; do
	echo
	echo "Uploading batch $line to Azure semifield-developed-images"
    prefix=${SAS%%?sv=*} # remove before ?sv=
    suffix=${SAS#*?sv=} # remove after   ?sv=
	dst=$prefix/$line?sv=$suffix
	jsondst=$prefix/$line/?sv=$suffix
	
	asfm="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images/$line/autosfm"
	meta_masks="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images/$line/meta_masks"
	metadata="/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images/$line/metadata"
		
	
	azcopy copy "${asfm-}" "${dst-}" --recursive
	azcopy copy "${meta_masks-}" "${dst-}" --recursive
	azcopy copy "${metadata-}" "${dst-}" --recursive
	echo "Done copying batch $line to Azure blob container semifield-developed-images"
	echo

done