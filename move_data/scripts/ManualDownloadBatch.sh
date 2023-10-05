#!/usr/bin/env bash

batchid=$1
item=$2
localdest="$3"
SAS=

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