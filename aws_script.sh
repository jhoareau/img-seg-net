#!/bin/bash

cd ..
git clone https://github.com/alexgkendall/SegNet-Tutorial.git
mv SegNet-Tutorial/CamVid CamVid
cd img-seg-net
mkdir /dev/xvdba/trainingdata
ln -s /dev/xvdba/trainingdata train
