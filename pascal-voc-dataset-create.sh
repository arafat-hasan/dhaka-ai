#!/bin/bash

# Use: ./dataset-create-symlink.sh "path/to/dhaka-ai/Final Train Dataset" "path/to/copy/ImageSet/Main"
# This script creates a dataset made of symlinks in Pascal VOC structure.
# Train-Val-Test split is also created.

set -e

if [ $# -eq 0 ]
  then
    echo "Supply data dir path"
    exit
fi

if [ -d datasets/voc/ ]; then
    rm -r datasets/voc/
fi


mkdir -p datasets/voc/JPEGImages
mkdir -p datasets/voc/Annotations
mkdir -p datasets/voc/ImageSets/Main

for jpg in "$1"/*.jpg
do
    ln -s "$jpg" datasets/voc/JPEGImages
done

for JPG in "$1"/*.JPG
do
    ln -s "$JPG" datasets/voc/JPEGImages
done


for jpeg in "$1"/*.jpeg
do
    ln -s "$jpeg" datasets/voc/JPEGImages
done

for png in "$1"/*.png
do
    ln -s "$png" datasets/voc/JPEGImages
done

for PNG in "$1"/*.PNG
do
    ln -s "$PNG" datasets/voc/JPEGImages
done


for xml in "$1"/*.xml
do
    ln -s "$xml"  datasets/voc/Annotations
done


python3 png2jpg.py  datasets/voc/JPEGImages/ # Convert all png files to jpg


rename JPG jpg datasets/voc/JPEGImages/*.JPG || :  # Convert uppercase file ext to lowercase
rename jpeg jpg datasets/voc/JPEGImages/*.jpeg || :

rename 's/\.JPG$/.jpg/' datasets/voc/JPEGImages/*.JPG || :
rename 's/\.jpeg$/.jpg/' datasets/voc/JPEGImages/*.jpeg || :


rm "datasets/voc/JPEGImages/231.jpg" # Corrupted file
rm "datasets/voc/Annotations/231.xml"    # Corrupted file

rm "datasets/voc/Annotations/Pias (359).xml"    # Corrupted file
rm "datasets/voc/JPEGImages/Pias (359).jpg" # Corrupted file

rm "datasets/voc/Annotations/Pias (360).xml"    # Corrupted file
rm "datasets/voc/JPEGImages/Pias (360).jpg" # Corrupted file


if [ $2 = "generate" ]
  then
      python3 generateimagesets.py    # Create train-test-val split in ImageSets dir  
      echo "Image Set Created"
else
    cp -r datasets/ImageSets/Main/* datasets/voc/ImageSets/Main/
    echo "Image Set Copied"
fi


echo "Done"
