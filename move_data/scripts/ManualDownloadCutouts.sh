#!/usr/bin/env bash

batchid=$1
localdest="$2"
SAS=

# Should only have 1 line to read
prefix=${SAS%%semifield-cutouts*} # remove after -
suffix=${SAS#*semifield-cutouts} # remove before _
azuresrc=${prefix-}semifield-cutouts/${batchid-}/${suffix-}

if [ ! -d "${localdest-}" ]; then
	mkdir -p "${localdest-}"
	echo "Directory made for ${localdest-}"
fi

if [ -d "${localdest-}" ]; then
	echo
	echo "Moving ${batchid-} from azure semifield-cutouts to ${localdest-} "
	azcopy copy "${azuresrc-}" "${localdest-}" --recursive --overwrite=false
	echo
fi