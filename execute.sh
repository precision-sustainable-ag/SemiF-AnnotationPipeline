#!/usr/bin/env bash

set -o nounset
set -o pipefail

COMMAND=./execute.sh

## Main processessing section that runs individual processes 
## but moves to another batch if any single process fails with exit code 1.

# Read unprocessed batches
need_process=./.batchlogs/unprocessed.txt

# while read line
count=0
IFS=$'\n'

for line in $(cat $need_process)
    do
        # echo $line
        new_line=${line%%:*}
        echo $new_line
        # run without finding unprocessed
        python PIPELINE.py general.batch_id=${new_line-}
    
    count=$((count+1))
        
    done