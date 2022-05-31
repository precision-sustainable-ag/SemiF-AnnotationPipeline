#!/usr/bin/env bash

AUTOSFM_BRANCH=$1

# https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides
# Absolute path to this script
SCRIPT=$(readlink -f "$0")
# Update the SemiF-Annotation repo
SEMIF_DIR=$(dirname "$SCRIPT")/..

if [ -f "${SEMIF_DIR}/control/execution" ]
then
  echo "Execution ongoing, aborting maintenance."
  exit 1
fi

if [ -z "${AUTOSFM_DIR}" ]
then
  echo "AUTOSFM_DIR must be set in the environment"
  exit 1
fi

touch "${SEMIF_DIR}/control/autosfmlock"

# Override the default branch with the branch provided
if [ $AUTOSFM_BRANCH ]
then
  echo "Overriding default autoSfM branch 'master' with ${AUTOSFM_BRANCH}"
else
  AUTOSFM_BRANCH="master"
  echo "Using defualt autoSfM branch '${AUTOSFM_BRANCH}'"
fi

# Update the AutoSfM repo
cd $AUTOSFM_DIR
# git pull origin $AUTOSFM_BRANCH
echo "A dummy autoSfM pull here"
# Make sure the docker container is up. Ideally, will not build again
docker build -t sfm .

rm "${SEMIF_DIR}/control/autosfmlock"

now=$(date +"%D %T")
echo "AutoSfM Docker container rebuilt on : $now"

touch "${SEMIF_DIR}/control/semifmlock"

echo "Changing directory to ${SEMIF_DIR}"
cd $SEMIF_DIR
# Make sure the code is up to date
# git pull origin master
echo "A dummy SemiF pull here"

rm "${SEMIF_DIR}/control/semifmlock"

now=$(date +"%D %T")
echo "Latest pull for SemiF-Annotation on : $now"
