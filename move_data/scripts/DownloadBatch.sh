#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
	echo
	echo 'Usage: bash DownloadBatch.sh batchid item dest sas_token

	    Downloads batch directories necessary for autoSfM (images, masks). 
		Uses batch ("batchid") and specific directory items ("item") for path information along with destination ("dest"). 
		Downloads from azure blob container. SAS token must be in double quotes

		"dest" must be a "data" directory

		Ex. bash DownloadBatch.sh NC_2022-08-05 images ./data/semifield-developed-images "<SAS token>"
	    '
	        exit
fi

batchid=$1
item=$2
localdest="$3/${batchid-}"
SAS=$4

# Should only have 1 line to read
prefix=${SAS%%semifield-developed-images*} # remove after -
suffix=${SAS#*semifield-developed-images} # remove before _
azuresrc=${prefix-}semifield-developed-images/${batchid-}/${item-}${suffix-}

if [ ! -d "${localdest-}" ]; then
	mkdir -p "${localdest-}"
	echo "Directory made for ${localdest-}"
fi

if [ -d "${localdest-}" ]; then
	echo
	echo "${batchid-} exists."
	echo "Moving ${item-} from azure to ${localdest-} "
	azcopy copy "${azuresrc-}" "${localdest-}" --recursive --overwrite=false
	echo
fi