#!/usr/bin/env bash

set -o nounset
set -o pipefail

COMMAND=./execute.sh
DONEYET="${COMMAND}.alreadyrun"


## Main processessing section that runs individual processes 
## but moves to another batch if any single process fails with exit code 1.

# Read unprocessed batches
need_process=./.batchlogs/unprocessed.txt
while read -r line
    do
        line=${line%%:*}
        # If exit code for the below command is not 1, then continue processing,
        # else skips remaining processes and goes to the next batch. 
        # This is done for each process.
            
            # Download images
        python PIPELINE.py general.batch_id=${line-} general.task=download_data
        if [  $? -eq 0 ]; then
            python PIPELINE.py general.batch_id=${line-} general.task=autosfm_pipeline
            if [  $? -eq 0 ]; then
                # AutoSfM
                python PIPELINE.py general.batch_id=${line-} general.task=localize_plants
                if [  $? -eq 0 ]; then
                    # Remap labels
                    python PIPELINE.py general.batch_id=${line-} general.task=remap_labels
                    if [  $? -eq 0 ]; then
                        # Assign Species
                        python PIPELINE.py general.batch_id=${line-} general.task=assign_species
                        if [  $? -eq 0 ]; then
                            # Segment Vegetation
                            python PIPELINE.py general.batch_id=${line-} general.task=segment_vegetation
                            if [  $? -eq 0 ]; then
                                # Upload data
                                python PIPELINE.py general.batch_id=${line-} general.task=upload_data
                                if [  $? -eq 0 ]; then
                                    # If all processes were successfully run without a 1 exit code, 
                                    # then the below section writes the batch to "prcessed.txt" and removes 
                                    # batch from "unprocessed.txt"
                                    echo ${line-} >> .batchlogs/processed.txt
                                    sed -i "/$line/d" "$need_process"  
                                fi
                            fi
                        fi
                    fi
                fi
            fi
        # Terminal message if a batch is being skipped. (ie exit code = 1)
        else
            echo
            echo "                          Warning: Something went wrong with ${line-}. Continuing with next batch."
            echo
        fi

    done < ${need_process-}