#!/usr/bin/env bash

function sigint_handler()
{
    echo "SIGINT detected, exiting with status 1. Removing execution file."
    rm control/executing
    exit 1
}

function sigterm_handler()
{
    echo "SIGTERM detected, exiting with status 1. Removing execution file."
    rm control/executing
    exit 1
}

trap 'kill -INT "${python_process_pid}"; wait "${python_process_pid}"; sigint_handler' SIGINT
trap 'kill -TERM "${python_process_pid}"; wait "${python_process_pid}"; sigterm_handler' SIGTERM


BATCH_ID=$1
METASHAPE_KEY=$2

# https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides
# Absolute path to this script
SCRIPT=$(readlink -f "$0")
# Update the SemiF-Annotation repo
SEMIF_DIR=$(dirname "$SCRIPT")/..

# Check if maintenance is being performed
if [ -f "${SEMIF_DIR}/control/autosfmlock" ] || [ -f "${SEMIF_DIR}/control/semiflock" ]
then
  echo "Maintenance ongoing, exiting without proccessing"
  exit 1
fi

# Run the SemiF-Pipeline
cd $SEMIF_DIR
# Create a execution file
touch control/executing

python SEMIF.py general.batch_id=$BATCH_ID autosfm.metashape_key=$METASHAPE_KEY &
python_process_pid="$!"
wait "$python_process_pid"

rm control/executing
