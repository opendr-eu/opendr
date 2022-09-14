#!/bin/bash

wget --load-cookies /tmp/cookies.txt "ftp://opendrdata.csd.auth.gr/simulation/human_data_generation_framework/human_models.tar.gz" -O human_models.tar.gz && rm -rf /tmp/cookies.txt

tar -xzvf human_models.tar.gz

rm human_models.tar.gz

mkdir -p ./background_images/Cityscapes/in
mkdir -p ./background_images/Cityscapes/out

wget --load-cookies /tmp/cookies.txt "ftp://opendrdata.csd.auth.gr/simulation/human_data_generation_framework/csv.tar.gz" -O csv.tar.gz && rm -rf /tmp/cookies.txt

tar -xzvf csv.tar.gz

rm csv.tar.gz

wget --load-cookies /tmp/cookies.txt "ftp://opendrdata.csd.auth.gr/simulation/human_data_generation_framework/img_ids.pkl" -O img_ids.pkl && rm -rf /tmp/cookies.txt


mv ./img_ids.pkl ./background_images/Cityscapes/img_ids.pkl

wget --load-cookies /tmp/cookies.txt "ftp://opendrdata.csd.auth.gr/simulation/human_data_generation_framework/human_colormap.txt" -O human_colormap.txt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "ftp://opendrdata.csd.auth.gr/simulation/human_data_generation_framework/locations_colormap.txt" -O locations_colormap.txt && rm -rf /tmp/cookies.txt


mv ./human_colormap.txt ./background_images/Cityscapes/human_colormap.txt
mv ./locations_colormap.txt ./background_images/Cityscapes/locations_colormap.txt
