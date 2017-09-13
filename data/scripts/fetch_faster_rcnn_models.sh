#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

NET=vgg16
FILE=voc_0712_80k-110k.tgz
# replace it with gs11655.sp.cs.cmu.edu if ladoga.graphics.cs.cmu.edu does not work
URL=http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/$NET/$FILE
#CHECKSUM=cb32e9df553153d311cc5095b2f8c340

if [ -f $FILE ]; then
    echo "File already exists."
    exit 0
fi

echo "Downloading $NET Faster R-CNN models"

wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
