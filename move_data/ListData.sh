#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# You can change these
newfile=.batchlogs/batchdata.txt
resultfile=.batchlogs/temp_batchdata.txt

# Leave this
filename=SAS_download_key.txt

if [ -f ${newfile-} ]; then 
    > ${newfile-}
else
    touch ${newfile-}
fi

if [ -f ${resultfile-} ]; then
    > ${resultfile-}
else
    touch ${resultfile-}
fi

while read line; do
	# Should only have 1 line to read
    
    # find top level contents
    # azcopy ls ${line-} | cut -d/ -f 1 | awk '!a[$0]++'
    
    # find directories with depth N
    N=2
    azcopy ls ${line-} | cut -d/ -f 1-${N} | awk '!a[$0]++' >> ${resultfile}
    done <$filename

file=${resultfile-}
while IFS=';' read -ra line; do
    string=${line-}
    prefix="INFO: "
    foo=${string#"$prefix"}
    # grep -rohP "${foo-}" . | sort -u
    # touch $newfile
    echo $foo >> ${newfile-}
   
    done <$file

sort -o ${newfile-} ${newfile-}
rm ${resultfile-}