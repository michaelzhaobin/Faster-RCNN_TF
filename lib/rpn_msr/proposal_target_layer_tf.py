# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import pdb

DEBUG = False

#rpn_rois: (num of left proposal,*5) blob[:,0]==0 blob[:,1:5]: x1,y1,x2,y2
#gt_boxes: [[11,22,33,44,0],[22,33,44,55,2]]boxes +classes
#classes: 21
def proposal_target_layer(rpn_rois, gt_boxes,_num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

    # Include ground-truth boxes in the set of candidate rois
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    # (2, 1)
    all_rois = np.vstack(
        (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )
    """
    add all_rois in the bottom:
    [[0,x1, y1, x2, y2]
     [0,x11,y11,x21,y21]]
    """

    # Sanity check: single batch only
    assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_imagea #128
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image) # 0.25*128 = 42

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)
    """
    all_rois: (num of left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
    gt_boxes: [[11,22,33,44,0],[22,33,44,55,2]]boxes +classes;
    fg_rois_per_image: 42
    _num_classes: 21
    """
    """
    (1)labels: the final classes of the ground truth correspounding to per pred box [num of finally left proposal,] 
        for ex: [9,15,15,15,9,9....]
    (2) rois: (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
          [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
    (3): num of finally left proposals * 4*21: [dx,dy,dw,dh] of 1 class
    (4): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
    """

    if DEBUG:
        print 'num fg: {}'.format((labels > 0).sum())
        print 'num bg: {}'.format((labels == 0).sum())
        _count += 1
        _fg_num += (labels > 0).sum()
        _bg_num += (labels == 0).sum()
        print 'num fg avg: {}'.format(_fg_num / _count)
        print 'num bg avg: {}'.format(_bg_num / _count)
        print 'ratio: {:.3f}'.format(float(_fg_num) / float(_bg_num))

    rois = rois.reshape(-1,5)
    labels = labels.reshape(-1,1)
    #the final classes of the ground truth correspounding to per pred box [num of finally left proposal,1] 
    #    for ex: [[9],[15],[15],[15],[9],[9]....]
    bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    # to ture or false

    return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights
"""
(1) the final classes of the ground truth correspounding to per pred box [num of finally left proposal,1] 
        for ex: [[9],[15],[15],[15],[9],[9]....]
(2) rois: (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
          [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
(3): num of finally left proposals * 4*21: [dx,dy,dw,dh] of 1 class in 21
(4): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
(5): num of finally left proposals * 4*21: [true,true,true,true] of 1 class in 21; [false, false, false, false] in left of the classes
"""

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    """
    (1): (num of left boxes, 5)
        [:,0]: the final classes of the ground truth correspounding to per pred box [num of finally left proposal,]
        [:,1:5]: (num of left box * 4)[dx,dy,dw,dh]
    (2) 21
    """

    clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    #clss.size: num of finally left proposals
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    # the index of background
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS #[1,1,1,1]
    return bbox_targets, bbox_inside_weights
"""
(1): num of finally left proposals * 4*21: [dx,dy,dw,dh] of 1 class
(2): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
"""


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""
    """
    rois: (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
          [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
    gt_boxes[gt_assignment[keep_inds, :4]:the coordinates of the ground truth boxes correspounding to per pred box [num of finally left proposal,4]
    labels: the final classes of the ground truth correspounding to per pred box [num of finally left proposal,] :
    """

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    #(num of left box * 4)[dx,dy,dw,dh]
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        #false
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)
#return (num of left boxes, 5)
"""
[:,0]: the final classes of the ground truth correspounding to per pred box [num of finally left proposal,]
[:,1:5]: (num of left box * 4)[dx,dy,dw,dh]
"""

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    """
    all_rois: (num of left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
    gt_boxes: [[11,22,33,44,0],[22,33,44,55,2]]boxes +classes;
    fg_rois_per_image: 42
    _num_classes: 21
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    """
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float 
    Returns: overlaps: (N, K) ndarray of overlap between boxes and query_boxes (N,2)
    """
    gt_assignment = overlaps.argmax(axis=1)
    #return the index of gt_box which has max overlap with pred_box [N, ] ex:[0,1,1,1,1,0,0,1]
    max_overlaps = overlaps.max(axis=1)
    ##return max overlap per pred box and two gt_box [N,]
    labels = gt_boxes[gt_assignment, 4]
    #return the classes of the ground truth correspounding to per pred box [N,] 
    #for ex: [9,15,15,15,9,9,,,,]

    # Select foreground RoIs as those with >= FG_THRESH overlap; return index of it
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size)) # 42; ..
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
        # choose index randomly, if fg_inds = 5,size =2; for example, return: [2,4]
        #return the index of finally left foregrounds

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    #__C.TRAIN.BG_THRESH_HI = 0.5; __C.TRAIN.BG_THRESH_LO = 0.1
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        #return the index of finally left backgrounds

    # The index that we finally select (both fg[0:fg_rois_per_this_image] and bg[fg_rois_per_this_image:])
    keep_inds = np.append(fg_inds, bg_inds)
    #return the final classes of the ground truth correspounding to per pred box [N,] :
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    #rois: (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
    #[0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    """
    rois: (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
          [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
    gt_boxes[gt_assignment[keep_inds, :4]:the coordinates of the ground truth correspounding to per pred box [num of finally left proposal,4]
    labels: the final classes of the ground truth correspounding to per pred box [num of finally left proposal,] :
    """
    #return (num of left boxes, 5)
    """
    [:,0]: the final classes of the ground truth correspounding to per pred box [num of finally left proposal,]
    [:,1:5]: (num of left box * 4)[dx,dy,dw,dh]
    """

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)
    """
    (1): (num of left boxes, 5)
        [:,0]: the final classes of the ground truth correspounding to per pred box [num of finally left proposal,]
        [:,1:5]: (num of left box * 4)[dx,dy,dw,dh]
    (2) 21
    """
    """
    return:
    (1): num of finally left proposals * 4*21: [dx,dy,dw,dh] of 1 class
    (2): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
    """

    return labels, rois, bbox_targets, bbox_inside_weights
"""
    (1)labels: the final classes of the ground truth correspounding to per pred box [num of finally left proposal,] 
        for ex: [9,15,15,15,9,9....]
    (2) rois: (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
          [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
    (3): num of finally left proposals * 4*21: [dx,dy,dw,dh] of 1 class
    (4): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
"""
