# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """
    #SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
    
    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG: # True
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # BBOX_RED: True; 
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        #rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, 
        #rpn_bbox_outside_weights)
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))
        #dot multiply; only let the anchor box of prop(box) = 1 exits, else are 0s

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        #tf.less: Returns the truth value of (x < y) element-wise (tensor of bool)
        # the elems of inside_mul which is < 1/9 are 1
        # the elems of inside_mul which is < 1/9 are 0
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul


    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        #RoIDataLayer(roidb, num_classes)

        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        # return dect(inputs)['rpn_cls_score_reshape']
        # rpn_cls_score_reshape: [1, 126(9*14),14,2]; output: [1764(9*14*14), 2]
        # 9: num of anchors
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        """
        rpn_labels:(1,1,14*9,14) elem: 1,0,-1(sum:14*14*9) including 128(1),128(0),left are -1(random choice of 256 for eliminiting biases)
        rpn_bbox_targets: 1, 9*4, 14, 14 elem: x move of center, y move of center, width transform , height transform
                  only the inside boxes are non-zero, the rest of boxes are [0 0 0 0]
        rpn_bbox_inside_weights: 1, 9*4, 14, 14 elem: when rpn_lables=1----[1.0,1,1,1]
                  only the inside boxes are non-zero, the rest of boxes are [0 0 0 0]
        rpn_bbox_outside_weights: 1, 9*4, 14, 14 elem: when rpn_lables=1 or 0----[1.0,1,1,1]/
                  only the inside boxes are non-zero, the rest of boxes are [0 0 0 0]
        """
        #return (14*14*9,)
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        #tf.not_equal: 返回逐个元素的布尔值; tf.where: 找出tensor里所有True值的index; 
        # tf.gather(params, indices, name = None): 根据indices索引,从params中取对应索引的值,然后返回
        # find the rows in rpn_cls_score which is useful
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        # find the rows in rpn_label which is useful
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
        """第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
        第二个参数labels：实际的标签，大小同上
        then: mean_value.
        """

       
        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        # rpn_bbox_pred:14*14*36 (9 anchors * 4) [1,14,14,36]
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        # rpn-data[1]: same as former-----[1,14,14,36]
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        # rpn-data[1]: same as former-----[1,14,14,36]
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])
        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
 

       
        # R-CNN classification loss
        cls_score = self.net.get_output('cls_score')
        # (num of final left proposals, 21)
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        # (num of final left proposals,)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
        # cls_score: (n)

        
        
        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]
        smooth_l1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))
        
        
        
        # final loss
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)

        
        
        # iintialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # get one batch
            blobs = data_layer.forward()
            """
            blobs['data']: [1,maxL,maxH,3]
            blobs['gt_boxes']:
                 [[11,22,33,44, 16]
                  [22,33,44,55, 8]
                                 ]boxes +classes
            blobs['im_info']
            [[max_length, max_width, im_scale]]
            im_scale: 缩放比例
            """

            # Make one SGD update
            feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes']}
            # self.net.data: shape=[None, None, None, 3]

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                #false
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
                                                                                                feed_dict=feed_dict,
                                                                                                options=run_options,
                                                                                                run_metadata=run_metadata)

            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                #false
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                # cfg.TRAIN.DISPLAY = 10
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                # SNAPSHOT_ITERS = 5000
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        # True
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        #True
        if cfg.IS_MULTISCALE:# false
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb
"""
imdb.roidb[i](an image example i)(2 objects in picture:person,cat):
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
def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        #true
        if cfg.IS_MULTISCALE:
            #false
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""
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
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps'] #[1,1]
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0] # 0.5 return[0,1]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # 0.1, 0.5 return [] vacuum
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    # train_net(network, imdb, roidb, output_dir, pretrained_model=data/pretrain_model/VGG_imagenet.npy , max_iters=args.max_iters)
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb) # barely no change
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
