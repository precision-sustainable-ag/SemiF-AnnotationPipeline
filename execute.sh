#!/usr/bin/env bash

set -o nounset
set -o pipefail

COMMAND=./execute.sh

## Main processessing section that runs individual processes 
## but moves to another batch if any single process fails with exit code 1.

# Read unprocessed batches
need_process=./.batchlogs/unprocessed.txt

# while read line
for line in `cat $need_process`
    do
        
        line=${line%%:*}

        python PIPELINE.py general.batch_id=${line-} 
    done
