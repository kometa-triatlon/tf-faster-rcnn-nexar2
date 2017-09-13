import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
from collections import namedtuple

from datasets.imdb import imdb
from model.config import cfg

from nexet_eval import Box, eval_detector

class nexet(imdb):
    def __init__(self, split):
        imdb.__init__(self, 'nexet_' + split)
        self._data_path = os.path.join(cfg.DATA_DIR, 'nexet')

        self._classes = ('__background__', 'bus', 'car', 'pickup_truck', 'truck', 'van')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_index = self._load_image_set_index(split)
        self._boxes_df = pd.read_csv(os.path.join(self._data_path, 'train_boxes.csv'))
        self._obj_proposer = 'gt'
        self._roidb_handler = self.gt_roidb
        self._config = {'iou_threshold': 0.75}

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        gt_boxes = self._get_gt_boxes()
        dt_boxes = self._get_boxes_from_list(all_boxes)
        print("Starting evaluation")
        print("Number of gt images: {}, number of output images: {}".format(len(gt_boxes), len(dt_boxes)))

        n_empty_gt_images = 0
        with open(os.path.join(output_dir, self.name + '_dt.csv'), 'w') as f:
            f.write('image_filename,x0,y0,x1,y1,label,confidence\n')
            for img_name, boxes in dt_boxes.iteritems():
                for b in boxes:
                    f.write("{},{},{},{},{},{},{}\n".format(img_name, b.x0, b.y0, b.x1, b.y1, b.label, b.confidence))

        with open(os.path.join(output_dir, self.name + '_gt.csv'), 'w') as f:
            f.write('image_filename,x0,y0,x1,y1,label,confidence\n')
            for img_name, boxes in gt_boxes.iteritems():
                for b in boxes:
                    f.write("{},{},{},{},{},{},{}\n".format(img_name, b.x0, b.y0, b.x1, b.y1, b.label, b.confidence))


        for img_obj in dt_boxes:
            if img_obj not in gt_boxes:
                print ('ERROR: detected image name not a ground truth image: {}'.format(img_obj))
                n_empty_gt_images += 1
                continue

        print('Number of empty GT images: {}'.format(n_empty_gt_images))
        print('Mean AP = {}'.format(eval_detector(gt_boxes, dt_boxes, self._config['iou_threshold'])))


    def _get_gt_boxes(self):
        """
        Read boxes for evaluation
        """

        boxes = dict()
        df = self._boxes_df[self._boxes_df['image_filename'].isin(self._image_index)]
        for img_name, indices in df.groupby('image_filename').groups.iteritems():
            boxes[img_name] = [Box(r['x0'], r['y0'], r['x1'], r['y1'], r['label'], r['confidence']) for _, r in self._boxes_df.loc[indices].iterrows()]
        return boxes

    def _get_boxes_from_list(self, all_boxes):
        boxes = dict()
        for cls in range(len(self._classes)):
            if cls == 0: continue
            for img_num, img_name in enumerate(self._image_index):
                if img_name in boxes:
                    boxes[img_name].extend([Box(r[0], r[1], r[2], r[3], str(cls), r[4]) for r in all_boxes[cls][img_num]])
                else:
                    boxes[img_name] = [Box(r[0], r[1], r[2], r[3], str(cls), r[4]) for r in all_boxes[cls][img_num]]
        return boxes


    def _load_image_set_index(self, split):
        image_set_file = os.path.join(self._data_path, split + '.csv')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        df = pd.read_csv(image_set_file)
        return df['image_filename'].tolist()


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'train', index)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_annotation(self, index):
        df = self._boxes_df.loc[self._boxes_df['image_filename'] == index]
        num_boxes = df.shape[0]

        if num_boxes > 0:
            boxes = df.as_matrix(['x0', 'y0', 'x1', 'y1'])
            gt_classes = np.zeros((num_boxes,), dtype=np.int32)
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.int32)
            seg_areas = np.zeros((num_boxes,), dtype=np.float32)

            index = 0
            for _, row in df.iterrows():
                cls = self._class_to_ind[row['label']]
                width = row['x1'] - row['x0']
                height = row['y1'] - row['y0']
                gt_classes[index] = cls
                overlaps[index, :] = [cls == i and 1 or 0 for i in range(self.num_classes)]
                seg_areas[index] = width * height
                index += 1

            return {'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': scipy.sparse.csr_matrix(overlaps),
                    'flipped': False,
                    'seg_areas': seg_areas }

        return { 'boxes': np.asarray([[0, 0, 10, 10]]),
                 'gt_classes': np.asarray([0]),
                 'gt_overlaps': scipy.sparse.csr_matrix(np.asarray([[ -1 for _ in range(self.num_classes)]])),
                 'flipped': False,
                 'seg_areas': np.asarray([1.0]) }


