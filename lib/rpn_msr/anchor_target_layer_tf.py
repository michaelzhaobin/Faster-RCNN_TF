# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import pdb

DEBUG = False
#input: [1,14,14,18]('rpn_cls_score'); [[11,22,33,44,0],[22,33,44,55,2]]boxes +classes('gt_boxes'); 
#       [[max_length, max_width, im_scale]]('im_info'); [1,maxL,maxH,3]('data')
#       _feat_stride = [16,]; anchor_scales = [8, 16, 32]
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, data, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    """
#      [[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

#      [[ -83.,  -39.,  100.,   56.],[-175.,  -87.,  192.,  104.],[-359., -183.,  376.,  200.],[ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],[-247., -247.,  264.,  264.],[ -35.,  -79.,   52.,   96.],[ -79., -167.,   96.,  184.],[-167., -343.,  184.,  360.]])
    """
    _num_anchors = _anchors.shape[0]
    # 9

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        print np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]
    #[max_length, max_width, im_scale]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]
    # 14,14

    if DEBUG:
        print 'AnchorTargetLayer: height', height, 'width', width
        print ''
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])
        print 'height, width: ({}, {})'.format(height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    #array([  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208])
    shift_y = np.arange(0, height) * _feat_stride
    #array([  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208])
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    """
    shift_x:array([[  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208],
       [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
        208]])
shift_y:
array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0],
       [ 16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,
         16],
       [ 32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,
         32],
       [ 48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,
         48],
       [ 64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,
         64],
       [ 80,  80,  80,  80,  80,  80,  80,  80,  80,  80,  80,  80,  80,
         80],
       [ 96,  96,  96,  96,  96,  96,  96,  96,  96,  96,  96,  96,  96,
         96],
       [112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
        112],
       [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
        128],
       [144, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144,
        144],
       [160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
        160],
       [176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176,
        176],
       [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
        192],
       [208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208,
        208]])
        """
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()
    """
    return [196,9,4]
    array([[  0,   0,   0,   0],
       [ 16,   0,  16,   0],
       [ 32,   0,  32,   0],
       [ 48,   0,  48,   0],
       [ 64,   0,  64,   0],
       [ 80,   0,  80,   0],
       [ 96,   0,  96,   0],
       [112,   0, 112,   0],
       [128,   0, 128,   0],
       [144,   0, 144,   0],
       [160,   0, 160,   0],
       [176,   0, 176,   0],
       [192,   0, 192,   0],
       [208,   0, 208,   0],
       [  0,  16,   0,  16],
       [ 16,  16,  16,  16],
       [ 32,  16,  32,  16],
       [ 48,  16,  48,  16],
       [ 64,  16,  64,  16],
       [ 80,  16,  80,  16],
       [ 96,  16,  96,  16],
       [112,  16, 112,  16],
       [128,  16, 128,  16],
       [144,  16, 144,  16],
       [160,  16, 160,  16],
       [176,  16, 176,  16],
       [192,  16, 192,  16],
       [208,  16, 208,  16],
       [  0,  32,   0,  32],
       [ 16,  32,  16,  32],
       [ 32,  32,  32,  32],
       [ 48,  32,  48,  32],
       [ 64,  32,  64,  32],
       [ 80,  32,  80,  32],
       [ 96,  32,  96,  32],
       [112,  32, 112,  32],
       [128,  32, 128,  32],
       [144,  32, 144,  32],
       [160,  32, 160,  32],
       [176,  32, 176,  32],
       [192,  32, 192,  32],
       [208,  32, 208,  32],
       [  0,  48,   0,  48],
       [ 16,  48,  16,  48],
       [ 32,  48,  32,  48],
       [ 48,  48,  48,  48],
       [ 64,  48,  64,  48],
       [ 80,  48,  80,  48],
       [ 96,  48,  96,  48],
       [112,  48, 112,  48],
       [128,  48, 128,  48],
       [144,  48, 144,  48],
       [160,  48, 160,  48],
       [176,  48, 176,  48],
       [192,  48, 192,  48],
       [208,  48, 208,  48],
       [  0,  64,   0,  64],
       [ 16,  64,  16,  64],
       [ 32,  64,  32,  64],
       [ 48,  64,  48,  64],
       [ 64,  64,  64,  64],
       [ 80,  64,  80,  64],
       [ 96,  64,  96,  64],
       [112,  64, 112,  64],
       [128,  64, 128,  64],
       [144,  64, 144,  64],
       [160,  64, 160,  64],
       [176,  64, 176,  64],
       [192,  64, 192,  64],
       [208,  64, 208,  64],
       [  0,  80,   0,  80],
       [ 16,  80,  16,  80],
       [ 32,  80,  32,  80],
       [ 48,  80,  48,  80],
       [ 64,  80,  64,  80],
       [ 80,  80,  80,  80],
       [ 96,  80,  96,  80],
       [112,  80, 112,  80],
       [128,  80, 128,  80],
       [144,  80, 144,  80],
       [160,  80, 160,  80],
       [176,  80, 176,  80],
       [192,  80, 192,  80],
       [208,  80, 208,  80],
       [  0,  96,   0,  96],
       [ 16,  96,  16,  96],
       [ 32,  96,  32,  96],
       [ 48,  96,  48,  96],
       [ 64,  96,  64,  96],
       [ 80,  96,  80,  96],
       [ 96,  96,  96,  96],
       [112,  96, 112,  96],
       [128,  96, 128,  96],
       [144,  96, 144,  96],
       [160,  96, 160,  96],
       [176,  96, 176,  96],
       [192,  96, 192,  96],
       [208,  96, 208,  96],
       [  0, 112,   0, 112],
       [ 16, 112,  16, 112],
       [ 32, 112,  32, 112],
       [ 48, 112,  48, 112],
       [ 64, 112,  64, 112],
       [ 80, 112,  80, 112],
       [ 96, 112,  96, 112],
       [112, 112, 112, 112],
       [128, 112, 128, 112],
       [144, 112, 144, 112],
       [160, 112, 160, 112],
       [176, 112, 176, 112],
       [192, 112, 192, 112],
       [208, 112, 208, 112],
       [  0, 128,   0, 128],
       [ 16, 128,  16, 128],
       [ 32, 128,  32, 128],
       [ 48, 128,  48, 128],
       [ 64, 128,  64, 128],
       [ 80, 128,  80, 128],
       [ 96, 128,  96, 128],
       [112, 128, 112, 128],
       [128, 128, 128, 128],
       [144, 128, 144, 128],
       [160, 128, 160, 128],
       [176, 128, 176, 128],
       [192, 128, 192, 128],
       [208, 128, 208, 128],
       [  0, 144,   0, 144],
       [ 16, 144,  16, 144],
       [ 32, 144,  32, 144],
       [ 48, 144,  48, 144],
       [ 64, 144,  64, 144],
       [ 80, 144,  80, 144],
       [ 96, 144,  96, 144],
       [112, 144, 112, 144],
       [128, 144, 128, 144],
       [144, 144, 144, 144],
       [160, 144, 160, 144],
       [176, 144, 176, 144],
       [192, 144, 192, 144],
       [208, 144, 208, 144],
       [  0, 160,   0, 160],
       [ 16, 160,  16, 160],
       [ 32, 160,  32, 160],
       [ 48, 160,  48, 160],
       [ 64, 160,  64, 160],
       [ 80, 160,  80, 160],
       [ 96, 160,  96, 160],
       [112, 160, 112, 160],
       [128, 160, 128, 160],
       [144, 160, 144, 160],
       [160, 160, 160, 160],
       [176, 160, 176, 160],
       [192, 160, 192, 160],
       [208, 160, 208, 160],
       [  0, 176,   0, 176],
       [ 16, 176,  16, 176],
       [ 32, 176,  32, 176],
       [ 48, 176,  48, 176],
       [ 64, 176,  64, 176],
       [ 80, 176,  80, 176],
       [ 96, 176,  96, 176],
       [112, 176, 112, 176],
       [128, 176, 128, 176],
       [144, 176, 144, 176],
       [160, 176, 160, 176],
       [176, 176, 176, 176],
       [192, 176, 192, 176],
       [208, 176, 208, 176],
       [  0, 192,   0, 192],
       [ 16, 192,  16, 192],
       [ 32, 192,  32, 192],
       [ 48, 192,  48, 192],
       [ 64, 192,  64, 192],
       [ 80, 192,  80, 192],
       [ 96, 192,  96, 192],
       [112, 192, 112, 192],
       [128, 192, 128, 192],
       [144, 192, 144, 192],
       [160, 192, 160, 192],
       [176, 192, 176, 192],
       [192, 192, 192, 192],
       [208, 192, 208, 192],
       [  0, 208,   0, 208],
       [ 16, 208,  16, 208],
       [ 32, 208,  32, 208],
       [ 48, 208,  48, 208],
       [ 64, 208,  64, 208],
       [ 80, 208,  80, 208],
       [ 96, 208,  96, 208],
       [112, 208, 112, 208],
       [128, 208, 128, 208],
       [144, 208, 144, 208],
       [160, 208, 160, 208],
       [176, 208, 176, 208],
       [192, 208, 192, 208],
       [208, 208, 208, 208]])
    """
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors #9
    K = shifts.shape[0] #196
    all_anchors = (_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    """
    return[196,9,4]
    array([[[ -83.,  -39.,  100.,   56.],
        [-175.,  -87.,  192.,  104.],
        [-359., -183.,  376.,  200.],
        ...,
        [ -35.,  -79.,   52.,   96.],
        [ -79., -167.,   96.,  184.],
        [-167., -343.,  184.,  360.]],

       [[ -67.,  -39.,  116.,   56.],
        [-159.,  -87.,  208.,  104.],
        [-343., -183.,  392.,  200.],
        ...,
        [ -19.,  -79.,   68.,   96.],
        [ -63., -167.,  112.,  184.],
        [-151., -343.,  200.,  360.]],

       [[ -51.,  -39.,  132.,   56.],
        [-143.,  -87.,  224.,  104.],
        [-327., -183.,  408.,  200.],
        ...,
        [  -3.,  -79.,   84.,   96.],
        [ -47., -167.,  128.,  184.],
        [-135., -343.,  216.,  360.]],

       ...,

       [[  93.,  169.,  276.,  264.],
        [   1.,  121.,  368.,  312.],
        [-183.,   25.,  552.,  408.],
        ...,
        [ 141.,  129.,  228.,  304.],
        [  97.,   41.,  272.,  392.],
        [   9., -135.,  360.,  568.]],

       [[ 109.,  169.,  292.,  264.],
        [  17.,  121.,  384.,  312.],
        [-167.,   25.,  568.,  408.],
        ...,
        [ 157.,  129.,  244.,  304.],
        [ 113.,   41.,  288.,  392.],
        [  25., -135.,  376.,  568.]],

       [[ 125.,  169.,  308.,  264.],
        [  33.,  121.,  400.,  312.],
        [-151.,   25.,  584.,  408.],
        ...,
        [ 173.,  129.,  260.,  304.],
        [ 129.,   41.,  304.,  392.],
        [  41., -135.,  392.,  568.]]])
    _anchors.reshape((1, 9, 4)):
    array([[[ -83.,  -39.,  100.,   56.],
        [-175.,  -87.,  192.,  104.],
        [-359., -183.,  376.,  200.],
        [ -55.,  -55.,   72.,   72.],
        [-119., -119.,  136.,  136.],
        [-247., -247.,  264.,  264.],
        [ -35.,  -79.,   52.,   96.],
        [ -79., -167.,   96.,  184.],
        [-167., -343.,  184.,  360.]]])
    shifts.reshape((1, 196, 4)).transpose((1, 0, 2):
    array([[[  0,   0,   0,   0]],

       [[ 16,   0,  16,   0]],

       [[ 32,   0,  32,   0]],

       [[ 48,   0,  48,   0]],

       [[ 64,   0,  64,   0]],

       [[ 80,   0,  80,   0]],

       [[ 96,   0,  96,   0]],

       [[112,   0, 112,   0]],

       [[128,   0, 128,   0]],

       [[144,   0, 144,   0]],

       [[160,   0, 160,   0]],

       [[176,   0, 176,   0]],

       [[192,   0, 192,   0]],

       [[208,   0, 208,   0]],

       [[  0,  16,   0,  16]],

       [[ 16,  16,  16,  16]],

       [[ 32,  16,  32,  16]],

       [[ 48,  16,  48,  16]],

       [[ 64,  16,  64,  16]],

       [[ 80,  16,  80,  16]],

       [[ 96,  16,  96,  16]],

       [[112,  16, 112,  16]],

       [[128,  16, 128,  16]],

       [[144,  16, 144,  16]],

       [[160,  16, 160,  16]],

       [[176,  16, 176,  16]],

       [[192,  16, 192,  16]],

       [[208,  16, 208,  16]],

       [[  0,  32,   0,  32]],

       [[ 16,  32,  16,  32]],

       [[ 32,  32,  32,  32]],

       [[ 48,  32,  48,  32]],

       [[ 64,  32,  64,  32]],

       [[ 80,  32,  80,  32]],

       [[ 96,  32,  96,  32]],

       [[112,  32, 112,  32]],

       [[128,  32, 128,  32]],

       [[144,  32, 144,  32]],

       [[160,  32, 160,  32]],

       [[176,  32, 176,  32]],

       [[192,  32, 192,  32]],

       [[208,  32, 208,  32]],

       [[  0,  48,   0,  48]],

       [[ 16,  48,  16,  48]],

       [[ 32,  48,  32,  48]],

       [[ 48,  48,  48,  48]],

       [[ 64,  48,  64,  48]],

       [[ 80,  48,  80,  48]],

       [[ 96,  48,  96,  48]],

       [[112,  48, 112,  48]],

       [[128,  48, 128,  48]],

       [[144,  48, 144,  48]],

       [[160,  48, 160,  48]],

       [[176,  48, 176,  48]],

       [[192,  48, 192,  48]],

       [[208,  48, 208,  48]],

       [[  0,  64,   0,  64]],

       [[ 16,  64,  16,  64]],

       [[ 32,  64,  32,  64]],

       [[ 48,  64,  48,  64]],

       [[ 64,  64,  64,  64]],

       [[ 80,  64,  80,  64]],

       [[ 96,  64,  96,  64]],

       [[112,  64, 112,  64]],

       [[128,  64, 128,  64]],

       [[144,  64, 144,  64]],

       [[160,  64, 160,  64]],

       [[176,  64, 176,  64]],

       [[192,  64, 192,  64]],

       [[208,  64, 208,  64]],

       [[  0,  80,   0,  80]],

       [[ 16,  80,  16,  80]],

       [[ 32,  80,  32,  80]],

       [[ 48,  80,  48,  80]],

       [[ 64,  80,  64,  80]],

       [[ 80,  80,  80,  80]],

       [[ 96,  80,  96,  80]],

       [[112,  80, 112,  80]],

       [[128,  80, 128,  80]],

       [[144,  80, 144,  80]],

       [[160,  80, 160,  80]],

       [[176,  80, 176,  80]],

       [[192,  80, 192,  80]],

       [[208,  80, 208,  80]],

       [[  0,  96,   0,  96]],

       [[ 16,  96,  16,  96]],

       [[ 32,  96,  32,  96]],

       [[ 48,  96,  48,  96]],

       [[ 64,  96,  64,  96]],

       [[ 80,  96,  80,  96]],

       [[ 96,  96,  96,  96]],

       [[112,  96, 112,  96]],

       [[128,  96, 128,  96]],

       [[144,  96, 144,  96]],

       [[160,  96, 160,  96]],

       [[176,  96, 176,  96]],

       [[192,  96, 192,  96]],

       [[208,  96, 208,  96]],

       [[  0, 112,   0, 112]],

       [[ 16, 112,  16, 112]],

       [[ 32, 112,  32, 112]],

       [[ 48, 112,  48, 112]],

       [[ 64, 112,  64, 112]],

       [[ 80, 112,  80, 112]],

       [[ 96, 112,  96, 112]],

       [[112, 112, 112, 112]],

       [[128, 112, 128, 112]],

       [[144, 112, 144, 112]],

       [[160, 112, 160, 112]],

       [[176, 112, 176, 112]],

       [[192, 112, 192, 112]],

       [[208, 112, 208, 112]],

       [[  0, 128,   0, 128]],

       [[ 16, 128,  16, 128]],

       [[ 32, 128,  32, 128]],

       [[ 48, 128,  48, 128]],

       [[ 64, 128,  64, 128]],

       [[ 80, 128,  80, 128]],

       [[ 96, 128,  96, 128]],

       [[112, 128, 112, 128]],

       [[128, 128, 128, 128]],

       [[144, 128, 144, 128]],

       [[160, 128, 160, 128]],

       [[176, 128, 176, 128]],

       [[192, 128, 192, 128]],

       [[208, 128, 208, 128]],

       [[  0, 144,   0, 144]],

       [[ 16, 144,  16, 144]],

       [[ 32, 144,  32, 144]],

       [[ 48, 144,  48, 144]],

       [[ 64, 144,  64, 144]],

       [[ 80, 144,  80, 144]],

       [[ 96, 144,  96, 144]],

       [[112, 144, 112, 144]],

       [[128, 144, 128, 144]],

       [[144, 144, 144, 144]],

       [[160, 144, 160, 144]],

       [[176, 144, 176, 144]],

       [[192, 144, 192, 144]],

       [[208, 144, 208, 144]],

       [[  0, 160,   0, 160]],

       [[ 16, 160,  16, 160]],

       [[ 32, 160,  32, 160]],

       [[ 48, 160,  48, 160]],

       [[ 64, 160,  64, 160]],

       [[ 80, 160,  80, 160]],

       [[ 96, 160,  96, 160]],

       [[112, 160, 112, 160]],

       [[128, 160, 128, 160]],

       [[144, 160, 144, 160]],

       [[160, 160, 160, 160]],

       [[176, 160, 176, 160]],

       [[192, 160, 192, 160]],

       [[208, 160, 208, 160]],

       [[  0, 176,   0, 176]],

       [[ 16, 176,  16, 176]],

       [[ 32, 176,  32, 176]],

       [[ 48, 176,  48, 176]],

       [[ 64, 176,  64, 176]],

       [[ 80, 176,  80, 176]],

       [[ 96, 176,  96, 176]],

       [[112, 176, 112, 176]],

       [[128, 176, 128, 176]],

       [[144, 176, 144, 176]],

       [[160, 176, 160, 176]],

       [[176, 176, 176, 176]],

       [[192, 176, 192, 176]],

       [[208, 176, 208, 176]],

       [[  0, 192,   0, 192]],

       [[ 16, 192,  16, 192]],

       [[ 32, 192,  32, 192]],

       [[ 48, 192,  48, 192]],

       [[ 64, 192,  64, 192]],

       [[ 80, 192,  80, 192]],

       [[ 96, 192,  96, 192]],

       [[112, 192, 112, 192]],

       [[128, 192, 128, 192]],

       [[144, 192, 144, 192]],

       [[160, 192, 160, 192]],

       [[176, 192, 176, 192]],

       [[192, 192, 192, 192]],

       [[208, 192, 208, 192]],

       [[  0, 208,   0, 208]],

       [[ 16, 208,  16, 208]],

       [[ 32, 208,  32, 208]],

       [[ 48, 208,  48, 208]],

       [[ 64, 208,  64, 208]],

       [[ 80, 208,  80, 208]],

       [[ 96, 208,  96, 208]],

       [[112, 208, 112, 208]],

       [[128, 208, 128, 208]],

       [[144, 208, 144, 208]],

       [[160, 208, 160, 208]],

       [[176, 208, 176, 208]],

       [[192, 208, 192, 208]],

       [[208, 208, 208, 208]]])
    """
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    #_allowed_border = 0; return the coordinates of row: [1, 6, 8]

    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors.shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    """boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    N = num of anchors
    k = real number od objects in a picture
    """
    """for example overlaps:
    array([[ -83.,  -39.,  100.,   56.],
       [-175.,  -87.,  192.,  104.],
       [-359., -183.,  376.,  200.],
       [ -55.,  -55.,   72.,   72.],
       [-119., -119.,  136.,  136.],
       [-247., -247.,  264.,  264.],
       [ -35.,  -79.,   52.,   96.],
       [ -79., -167.,   96.,  184.],
       [-167., -343.,  184.,  360.]])
    """
    argmax_overlaps = overlaps.argmax(axis=1)
    # 返回沿轴axis最大值的索引 (N,) max overlapping class(groundtruth boxes) for every anchor 
    # array([2, 2, 2, 2, 2, 2, 3, 3, 3])
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    # 返回沿轴axis最大值 (N,) overlap between that anchor and corresponding max overlapping class every anchpor 
    # array([100., 192., 376.,  72., 136., 264.,  96., 184., 360.])
    gt_argmax_overlaps = overlaps.argmax(axis=0) 
    # 返回沿轴axis最大值的索引 (k ,) or (2, ) max overlapping anchor for every class
    # array([6, 0, 2, 8])
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])] 
    # 返回沿轴axis最大值 (N,) or (2,) overlap between that class and corresponding max overlapping anchor every class 
    # array([-35., -39., 376., 360.])
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    # array([0, 2, 6, 8])
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # false
        # assign bg labels first so that positive labels can clobber them (background)
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 # 0.3
        
    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU 0.7
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    # Max number of foreground examples(0.5) *  Total number of examples(256)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    #[[11,22,33,44,0],[22,33,44,55,2]]boxes +classes('gt_boxes')
    #argmax_overlaps: max overlapping class(groundtruth boxes) for every anchor (N(num of anchors),) exa:[0 0 1 0 1 1 1 1 1]
    """
    anchors:
    [[ 125.,  169.,  308.,  264.],
     [  33.,  121.,  400.,  312.],
     [-151.,   25.,  584.,  408.],
     [ 173.,  129.,  260.,  304.],
     [ 129.,   41.,  304.,  392.],
     [  41., -135.,  392.,  568.]
     [ 173.,  129.,  260.,  304.],
     [ 129.,   41.,  304.,  392.],
     [  41., -135.,  392.,  568.]]
    gt_boxes[argmax_overlaps, :]:
    [[11,22,33,44,0],
    [11,22,33,44,0]
    [22,33,44,55,2]
    [11,22,33,44,0]
    [22,33,44,55,2]
    [22,33,44,55,2]
    [22,33,44,55,2]
    [22,33,44,55,2]
    [22,33,44,55,2]]
    return:
    num of anchors(N) * 4
    x move of center, y move of center, width transform , height transform
    """
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
    # RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        #TRAIN.RPN_POSITIVE_WEIGHT = -1.0
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        # useful box numbers 256
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    # (num of total anchors,)
    #total_anchors = 196*9; ins_inside: num of inside anchors ;labels: (num of inside anchors, )
    #0,1,-1
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    # num of total anchors * 4
    # x move of center, y move of center, width transform , height transform
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    # num of total anchors * 4
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    # num of total anchors * 4

    if DEBUG:
        #false
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    #pdb.set_trace()
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    # 1 14 14 9------1,9,14,14
    labels = labels.reshape((1, 1, A * height, width))
    # ------(1,1,14*9,14)
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    # 196*9,4------1,14,14,9*4-------1, 9*4, 14, 14

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #96*9,4------1,14,14,9*4-------1, 9*4, 14, 14
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #96*9,4------1,14,14,9*4-------1, 9*4, 14, 14
    #assert bbox_outside_weights.shape[2] == height
    #assert bbox_outside_weights.shape[3] == width

    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights
"""
rpn_labels:(1,1,14*9,14) elem: 1,0,-1
rpn_bbox_targets: 1, 9*4, 14, 14 elem: x move of center, y move of center, width transform , height transform
rpn_bbox_inside_weights: 1, 9*4, 14, 14 elem: when rpn_lables=1----[1.0,1,1,1]
rpn_bbox_outside_weights: 1, 9*4, 14, 14 elem: when rpn_lables=1 or 0----[1.0,1,1,1]/256
"""


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
