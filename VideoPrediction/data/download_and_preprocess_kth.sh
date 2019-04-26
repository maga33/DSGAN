#!/usr/bin/env bash

# exit if any command fails
set -e

TARGET_DIR=./data/kth
mkdir -p ${TARGET_DIR}/processed
mkdir -p ${TARGET_DIR}/raw
echo "Downloading kth dataset (this takes a while)"
URL=http://www.cs.nyu.edu/~denton/datasets/kth.tar.gz
wget $URL -P $TARGET_DIR/processed
tar -zxvf $TARGET_DIR/processed/kth.tar.gz -C $TARGET_DIR/processed/
rm $TARGET_DIR/processed/kth.tar.gz

for c in walking jogging running handwaving handclapping boxing
do
  URL=http://www.nada.kth.se/cvap/actions/"$c".zip
  wget $URL -P $TARGET_DIR/raw
  mkdir $TARGET_DIR/raw/$c
  unzip $TARGET_DIR/raw/"$c".zip -d $TARGET_DIR/raw/$c
  rm $TARGET_DIR/raw/"$c".zip
done
python data/convert_kth.py
python video_prediction/datasets/kth_dataset_emily.py

rm -rf ${TARGET_DIR}/raw
rm -rf ${TARGET_DIR}/processed
echo "Succesfully finished downloading and preprocessing dataset kth"

