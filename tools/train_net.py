#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str) # gpu
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int) #0
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str) # None
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int) #70000
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str) # data/pretrain_model/VGG_imagenet.npy
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str) # experiments/cfgs/faster_rcnn_end2end.yml
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str) # voc_2007_trainval
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true') # false
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str) #VGGnet_train
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER) # None

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        # experiments/cfgs/faster_rcnn_end2end.yml
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # false
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
    imdb = get_imdb(args.imdb_name)
    # imdb_name: voc_2007_trainval
    # return the class of pascel_voc database：datasets.pascal_voc(trainval, 2007)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)
    """
roidb[i](an image example i)(2 objects in picture:person,cat):
    {
    'boxes':
    [[23,34,54,32],
     [432,45,6,43]]
    'gt_classes': 
    [16,8] #the number corresponding to person and cat
    'gt_overlaps':
    (0 15)  1
    (1,7)   1
    'flipped':
    False(or True)
    seg_areas:
    [432,53]
    'image':image_path
    'widthr': width of a image
    'height': heigth of a image
    'max_classes': [15, 7]
    'max_overlaps': [1,1]
"""

    output_dir = get_output_dir(imdb, None)
    ##./output/default/voc_2017_trainval; where the log file save
    print 'Output will be saved to `{:s}`'.format(output_dir)

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    #/gpu:0
    print device_name

    network = get_network(args.network_name)
    #VGGnet_train
    print 'Use network `{:s}` in training'.format(args.network_name)

    train_net(network, imdb, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
    # pretrained model： data/pretrain_model/VGG_imagenet.npy
