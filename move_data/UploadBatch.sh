#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
	        echo
		echo 'Usage: bash UploadBatch.sh batchid item src
			
			batchid: 	batch id
			item:		subfolder read to upload to azure
			src: 		parent directory of batch folder (blob container placeholder)
			
			Uploads a single batch ("batchid") to the semifield-developed-image blob container.
			overwrite=false

			Ex. bash UploadBatch.sh NC_2022-08-05 autosfm SemiF-AnnotationPipeline/data/semifield-developed-images
									              '
										                      exit
fi

localsrc=$1
data=$2

 filename=SAS_upload_key.txt

while read line; do
# Should only have 1 line to read
    # Appends data to key destination
	prefix=${line%%semifield-developed-images*}
    suffix=${line#*semifield-developed-images} 
    dest=${prefix-}semifield-developed-images/${data-}${suffix-}
	
	# Exit scripts if trying to overwrite images, masks, or plant-detectionss
	if [ -d "$localsrc" ]; then
    	echo "Local source: $localsrc is a directory."
		echo "Saving to:"
		echo $dest
		echo
		azcopy copy "${localsrc-}" "${dest-}" --recursive
	
	elif [ -f "$localsrc" ]; then
		echo
    	echo "Local source: $localsrc is a file."
		echo "Destination:"
		echo $dest
		echo
		azcopy copy "${localsrc-}" "${dest-}"
	else
		echo
		echo "$localsrc is not a directory or file. Canceling upload."
		echo 
		exit 1
	fi
	
	## Save logs
	# Only save logs if the "data" is a Json file (the last file to be uploaded).
	# This avoids having to save logs for each dir upload.
	
	if [[ $data == *"json"* ]]; then
		# Redefine data vairable from base name 
		# and define batch log path
		data="$(dirname "${data}")"
		logsrc=.batchlogs/test/${data-}/*
		# Reappend data to key destination
		logdest=${prefix-}semifield-developed-images/${data-}/logs${suffix-}
		echo "Saving logs"
		azcopy copy "${logsrc-}" "${logdest-}"
	
	fi
	
	done <$filename
