#!/usr/bin/env bash

bash copy_batch_to_longterm.sh ../../.batchlogs/upload_batches.txt
bash copy_cutouts_to_longterm.sh ../../.batchlogs/upload_batches.txt
bash upload_batch_from_list.sh ../../.batchlogs/upload_batches.txt
bash upload_cutout_from_list.sh ../../.batchlogs/upload_batches.txt
