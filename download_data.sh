#!/bin/bash

mkdir data
cd data
mkdir flickr_30kentities
cd flickr_30kentities

wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar
wget https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip
wget https://github.com/BryanPlummer/flickr30k_entities/blob/master/train.txt
wget https://github.com/BryanPlummer/flickr30k_entities/blob/master/test.txt
wget https://github.com/BryanPlummer/flickr30k_entities/blob/master/val.txt
wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz


tar -xzf flickr30k.tar.gz
tar -xvf flickr30k-images.tar
unzip annotations.zip

rm flickr30k-images.tar
rm annotations.zip
rm flickr30k.tar.gz

mkdir annotations_flickr
mv Annotations annotations_flickr/
mv Sentences annotations_flickr/


cd ..
mkdir mscoco
cd mscoco

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

mkdir images
mv train2014.zip images/
mv val2014.zip images/


unzip images/train2014.zip
rm images/train2014.zip 

unzip images/val2014.zip
rm images/val2014.zip

unzip annotations_trainval2014.zip
rm annotations_trainval2014.zip

cd ../..