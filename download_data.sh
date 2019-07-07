#!/bin/bash

mkdir data
cd data
mkdir flickr_30kentities
cd flickr_30kentities

wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar
wget https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip
wget https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/train.txt
wget https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/test.txt
wget https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/val.txt
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

cd ..

mkdir visual_genome
cd visual_genome

wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
wget https://visualgenome.org/static/data/dataset/image_data.json.zip
wget https://visualgenome.org/static/data/dataset/region_descriptions.json.zip

unzip images.zip
unzip images2.zip
unzip image_data.json.zip
unzip region_descriptions.json.zip

mkdir images
mv VG_100K_2/* images/
mv VGG_100K/* images/

rm -r VGG_100K/
rm -r VGG_100K_2/
rm image_data.json.zip
rm region_descriptions.json.zip
rm images.zip
rm images2.zip

cd ..




