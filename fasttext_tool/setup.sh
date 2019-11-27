#!/bin/bash
if [ ! -d "fastText-0.9.1" ]; then
wget https://github.com/facebookresearch/fastText/archive/v0.9.1.zip
unzip v0.9.1.zip
fi
cd ./fastText-0.9.1
if [ ! -d "build" ]; then
make
sudo python3 setup.py install
fi