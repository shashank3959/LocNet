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