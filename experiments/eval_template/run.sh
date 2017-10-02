#!/bin/bash -x

ROOT_DIR=../..
NET=res101
TRAIN_DATASET=nexet_train
ITER=465000
MODEL=$ROOT_DIR/output/$NET/$TRAIN_DATASET/default/${NET}_faster_rcnn_iter_${ITER}.ckpt
IMGDIR=$ROOT_DIR/data/nexet/train
NDX=$ROOT_DIR/data/nexet/val.csv
CONF_THRESH=0.1
NUM_CLASSES=6
CLS_INDICES="1 2 3 4 5"
CLS_NAMES="bus car pickup_truck truck van"

./detect.py --net $NET --model $MODEL --imgdir $IMGDIR --index $NDX --outfile nexet_val_dt_${NET}_${TRAIN_DATASET}_${ITER}.csv \
            --num_classes $NUM_CLASSES \
            --class_names $CLS_NAMES \
            --class_indices $CLS_INDICES \
            --confidence_threshold $CONF_THRESH
