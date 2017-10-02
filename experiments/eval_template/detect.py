#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join('..','..', 'lib'))
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16
from utils.timer import Timer
import pandas as pd

# Print iterations progress
def progressbar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percent, '%', suffix))
    sys.stdout.flush()


def out_detections(outf, img_id, dets, cls_name, thresh):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0: return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        outf.write("%s,%.2f,%.2f,%.2f,%.2f,%s,%.4f\n" %( img_id, xmin, ymin, xmax, ymax, cls_name, score))


def annotate(sess, net, index, imgdir, outfilename, cls_indices, cls_names, conf_thresh, nms_thresh):
    """Detect object classes in an image using pre-computed object proposals."""

    total = len(index)
    progressbar(0, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
    outf = open(outfilename, 'w')
    outf.write('image_filename,x0,y0,x1,y1,label,confidence\n')
    for i, img_id in enumerate(index):
        img_file = os.path.join(imgdir, img_id)

        if not os.path.isfile(img_file):
            sys.stderr.write('{} does not exist\n'.format(img_file))
            continue
        img = cv2.imread(img_file, -1)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(sess, net, img)
        timer.toc()

        for cls_name, cls_ind in zip(cls_names, cls_indices):
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, nms_thresh)
            dets = dets[keep, :]
            out_detections(outf, img_id, dets, cls_name, conf_thresh)

        progressbar(i + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Object detection tool')
    parser.add_argument('--net', required=True, help='Network type (vgg16/res101)')
    parser.add_argument('--model', required=True, help='Path to the object detection model')
    parser.add_argument('--imgdir', required=True, help='Dir with images')
    parser.add_argument('--index', required=True)
    parser.add_argument('--outfile', required=True)
    parser.add_argument('--num_classes', required=True, type=int, help='The number of classes in the model')
    parser.add_argument('--class_names', required=True, type=str, nargs='+', help='Names of the classes to detect')
    parser.add_argument('--class_indices', type=int, nargs='+', help='Indices of the classes to detect')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='NMS threshold')
    parser.add_argument('--anchor_scales', type=int, nargs='+', default=[8, 16, 32])

    args = parser.parse_args()

    if not os.path.isfile(args.model + '.meta'):
        raise IOError(('{:s} not found.\n').format(args.model + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'vgg16':
        net = vgg16()
    else:
        raise NotImplementedError


    net.create_architecture("TEST", args.num_classes, tag='default',
                            anchor_scales=args.anchor_scales,
                            anchor_ratios=[0.5, 1, 2])

    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    print('Loaded network {:s}'.format(args.model))

    index = pd.read_csv(args.index)['image_filename'].tolist()
    annotate(sess, net, index, args.imgdir, args.outfile, args.class_indices, args.class_names, args.confidence_threshold, args.nms_threshold)

