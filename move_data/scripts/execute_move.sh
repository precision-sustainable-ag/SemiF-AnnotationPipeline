#!/usr/bin/env bash

LOGFILE="./pipeline.log"

echo "Starting pipeline at $(date)" >> $LOGFILE

bash copy_batch_to_longterm.sh ../../.batchlogs/upload_batches.txt >> $LOGFILE 2>&1
bash copy_cutouts_to_longterm.sh ../../.batchlogs/upload_batches.txt >> $LOGFILE 2>&1
bash upload_batch_from_list.sh ../../.batchlogs/upload_batches.txt >> $LOGFILE 2>&1
bash upload_cutout_from_list.sh ../../.batchlogs/upload_batches.txt >> $LOGFILE 2>&1

echo "Pipeline finished at $(date)" >> $LOGFILE