#!/bin/bash

# Use: ./dataset-create-symlink.sh "path/to/dhaka-ai/Final Train Dataset" "path/to/copy/ImageSet/Main"
# This script creates a dataset made of symlinks in Pascal yolo structure.
# Train-Val-Test split is also created.

set -e

if [ $# -eq 0 ]
then
  echo "Supply data dir path"
  exit
fi

if [ -d datasets/yolo/ ]; then
  rm -r datasets/yolo/
fi

imagesTMP=datasets/yolo/imagesTMP/

mkdir -p $imagesTMP


for jpg in "$1"/*.jpg
do
  ln -s "$jpg" $imagesTMP
done

for JPG in "$1"/*.JPG
do
  ln -s "$JPG" $imagesTMP
done

for jpeg in "$1"/*.jpeg
do
  ln -s "$jpeg" $imagesTMP
done

for png in "$1"/*.png
do
  ln -s "$png" $imagesTMP
done

for PNG in "$1"/*.PNG
do
  ln -s "$PNG" $imagesTMP
done




python3 png2jpg.py $imagesTMP  # Convert all png files to jpg


rename JPG jpg "$imagesTMP"/*.JPG || :  # Convert uppercase file ext to lowercase
rename jpeg jpg "$imagesTMP"/*.jpeg || :

rename 's/\.JPG$/.jpg/' "$imagesTMP"/*.JPG || :
rename 's/\.jpeg$/.jpg/' "$imagesTMP"/*.jpeg || :


rm "$imagesTMP"/231.jpg # Corrupted file
rm "datasets/yolo/imagesTMP/Pias (359).jpg" # Corrupted file
rm "datasets/yolo/imagesTMP/Pias (360).jpg" # Corrupted file


mkdir -p datasets/yolo/images/train
mkdir -p datasets/yolo/images/val
mkdir -p datasets/yolo/labels/train
mkdir -p datasets/yolo/labels/val


if [ "$2" = "generate" ]
then
  mkdir -p datasets/yolo/labelsTMP
  cp -r datasets/yololabels/labels/train/* datasets/yolo/labelsTMP/
  cp -r datasets/yololabels/labels/val/* datasets/yolo/labelsTMP/
  python3 yolo-train-val-split.py
  echo "New Train Val split generated."
  rm -r datasets/yolo/labelsTMP
else
  for filename in datasets/yololabels/labels/val/*.txt; do
    name=$(basename "$filename" .txt)
    mv "$imagesTMP""$name".jpg  datasets/yolo/images/val
  done
  for filename in datasets/yololabels/labels/train/*.txt; do
    name=$(basename "$filename" .txt)
    mv "$imagesTMP""$name".jpg  datasets/yolo/images/train
  done
  echo "Previous train-val split used."
fi



rm -r $imagesTMP



echo "Done"
