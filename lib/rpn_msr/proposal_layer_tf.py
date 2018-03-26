# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
import pdb


DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""
#input: 'rpn_cls_prob_reshape'(1,14,14,18)(2 score values after softmax),'rpn_bbox_pred'(1,14,14,36)(dx,dx,dw,dh),
#'im_info': [[max_length, max_width, im_scale]]
# _feat_stride:[16,];  anchor_scales:[8,16,32]; cfg_key = 'TRAIN'
def proposal_layer(rpn_cls_prob_reshape,rpn_bbox_pred,im_info,cfg_key,_feat_stride = [16,],anchor_scales = [8, 16, 32]):
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)
    _anchors = generate_anchors(scales=np.array(anchor_scales)) # return 9*4 anchors coordinates
    _num_anchors = _anchors.shape[0]
    rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2]) #2 score values after softmax------(1,18,14,14)
    rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2]) #------(1,36,14,14)
    #rpn_cls_prob_reshape = np.transpose(np.reshape(rpn_cls_prob_reshape,[1,rpn_cls_prob_reshape.shape[0],rpn_cls_prob_reshape.shape[1],rpn_cls_prob_reshape.shape[2]]),[0,3,2,1])
    #rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,2,1])
    im_info = im_info[0] # [max_length, max_width, im_scale]

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
    #cfg_key = 'TEST'
    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N 
    #12000 Number of top scoring boxes to keep before apply NMS to RPN proposals
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    #2000 Number of top scoring boxes to keep after applying NMS to RPN proposals
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
    #0.7 NMS threshold used on RPN proposals
    min_size      = cfg[cfg_key].RPN_MIN_SIZE
    #16 Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)

    # the first set of _num_anchors channels are bg probs(background)
    # the second set are the fg probs, which we want
    scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :] #-------(1,9,14,14)
    bbox_deltas = rpn_bbox_pred
    #im_info = bottom[2].data[0, :]

    if DEBUG:
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])

    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:] #14; 14

    if DEBUG:# false
        print 'score map size: {}'.format(scores.shape)

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride #array([  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208])
    shift_y = np.arange(0, height) * _feat_stride #array([  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208])
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
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
        [  41., -135.,  392.,  568.]]])"""
    anchors = anchors.reshape((K * A, 4)) #(14*14*9, 4)

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) (1,36,14,14)format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))# (1,36,14,14)-----(1,14,14,36)------(14*14*9,4)
  
    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))# (1,9,14,14)-----(1,14,14,9)-----(14*14*9,1)

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)
    # generate the prodicted box（x1,y1,x2,y2） maped in original image (14*14*9, 4), only inside anchors, 
    # 因为bbox_deltas(dx,dy,dw,dh)是相对于相应anchors（x1,y1,x2,y2）的偏移

    # 2. clip predicted boxes to image boundaries
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size * im_info[2])
    #min_size:16; im_info[2]: max_width; """Remove all boxes with any side smaller than min_size. return index
    proposals = proposals[keep, :] #len(keep)*4
    scores = scores[keep] #len(keep)

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        #12000
        order = order[:pre_nms_topN]
    proposals = proposals[order, :] #len(order)*4
    scores = scores[order] #len(order)

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    # nms_thresh: 0.7
    if post_nms_topN > 0:
        #2000
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :] #<=2000
    scores = scores[keep]
    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32) #(2000, 1)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob
"""
return (num of left proposal(around 2000),*5)
blob[:,0]==0
blob[:,1:5]: x1,y1,x2,y2(in the original image)
"""
    #top[0].reshape(*(blob.shape))
    #top[0].data[...] = blob

    # [Optional] output scores blob
    #if len(top) > 1:
    #    top[1].reshape(*(scores.shape))
    #    top[1].data[...] = scores

def _filter_boxes(boxes, min_size):
    #min_size:16; im_info[2]: max_width; 
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
