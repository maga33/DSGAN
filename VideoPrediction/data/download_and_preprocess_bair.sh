#!/usr/bin/env bash

# exit if any command fails
set -e

TARGET_DIR=./data/bair
mkdir -p ${TARGET_DIR}
TAR_FNAME=bair_robot_pushing_dataset_v0.tar
URL=http://rail.eecs.berkeley.edu/datasets/${TAR_FNAME}
echo "Downloading bair dataset (this takes a while)"
wget ${URL} -O ${TARGET_DIR}/${TAR_FNAME}
tar -xvf ${TARGET_DIR}/${TAR_FNAME} --strip-components=1 -C ${TARGET_DIR}
rm ${TARGET_DIR}/${TAR_FNAME}
mkdir -p ${TARGET_DIR}/val
# reserve a fraction of the training set for validation
mv ${TARGET_DIR}/train/traj_{256_to_511,512_to_767,768_to_1023,1024_to_1279,1280_to_1535,1536_to_1791,1792_to_2047,2048_to_2303,2304_to_2559}.tfrecords ${TARGET_DIR}/val/
echo "Succesfully finished downloading and preprocessing dataset bair"
