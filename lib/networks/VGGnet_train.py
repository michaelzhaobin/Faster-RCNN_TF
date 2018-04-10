import tensorflow as tf
from networks.network import Network


#define

n_classes = 21
_feat_stride = [16,]
#14*16 = 224
anchor_scales = [8, 16, 32]

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        # shape：[1,maxL,maxH,3]
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        # [[max_length, max_width, im_scale]]
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        #[[11,22,33,44, 16]
        # [22,33,44,55, 8]]boxes +classes
        self.keep_prob = tf.placeholder(tf.float32)
        # 0.5
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        (self.feed('data')
         # input: 224*224*3 
         # this layer only change the self.inputs 
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         #output: 224*224*64
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
         #output: 224*224*64
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         #output: 112*112*64
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
         #output: 112*112*128
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
         #output: 112*112*128
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         #output: 56*56*128
             .conv(3, 3, 256, 1, 1, name='conv3_1')
         #output: 56*56*256
             .conv(3, 3, 256, 1, 1, name='conv3_2')
         #output: 56*56*256
             .conv(3, 3, 256, 1, 1, name='conv3_3')
         #output: 56*56*256
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         #output: 28*28*256
             .conv(3, 3, 512, 1, 1, name='conv4_1')
         #output: 28*28*512
             .conv(3, 3, 512, 1, 1, name='conv4_2')
         #output: 28*28*512
             .conv(3, 3, 512, 1, 1, name='conv4_3')
         #output: 28*28*512
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         #output: 14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_1')
         #output: 14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_2')
         #output: 14*14*512
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #output: 14*14*512
        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
         #output: 14*14*512
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))
        #output: 14*14*18 (9 anchors * 2 object scores) [1,14,14,18] tensorflow的表示方式
        
        # 'rpn_cls_score':[1,14,14,18]; 'gt_boxes': [[11,22,33,44, 16],[22,33,44,55, 8]]boxes +classes; 
        # 'im_info': [[max_length, max_width, im_scale]]; ['data']: [1,maxL,maxH,3]
        (self.feed('rpn_cls_score','gt_boxes','im_info','data')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))#<<<<<<<<<<<<<<<<<<
         """
        rpn_labels:(1,1,14*9,14) elem: 1,0,-1(sum:14*14*9) including 128(1)(fg_anchors),128(0)(bg_anchors)(how to choose is in paper),left are -1(random choice of 256 for eliminiting biases)
                   256 of inside of them is the after-choose where the value 1 represent fg_anchors and the value
                   0 represent bg_anchors; the rest of them is -1,which will be not considered
             how to choose: anchor交叠大于0.7某个阈值为1，交叠小于0.5为0，多了的随机选256个
        rpn_bbox_targets: 1, 9*4, 14, 14 elem: x move of center, y move of center, width transform , height transform(anchors relative to gt)
                  (only the inside boxes,but almost the same size as all anchors), the rest of boxes are [0 0 0 0]
        rpn_bbox_inside_weights: 1, 9*4, 14, 14 elem: the inside anchors are:[1,1,1,1] for left fg_anchors(labels == 1， around 128 个)
                  (only the inside boxes), the rest of boxes are [0 0 0 0]
        rpn_bbox_outside_weights: 1, 9*4, 14, 14 elem: the inside anchors are:[1/128,1/128,1/128,1/128] for left fg_bg_anchors(labels == 1 or 0， 256个)
                  (only the inside boxes), the rest of boxes are [0 0 0 0]
        """

        # Loss of rpn_cls & rpn_boxes
        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred')) #<<<<<<<<<<<<<<<<<
        # output: 14*14*36 (9 anchors * 4) [1,14,14,36](dx,dx,dw,dh)
        # 该结果是相对于相应anchors的dx,dx,dw,dh

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')#<<<<<<<<<<<<<<<<<<
        # output: [1, 126(9*14),14,2]
             .softmax(name='rpn_cls_prob')
        # output: [1, 126(9*14),14,2]

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))
        # output: (1,14,14,18)
       

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois')))
        # choose pred_box by totally the predicted box coordinates and scores itself； 
        # im_info: [[max_length, max_width, im_scale]]
        # 选择在原图内部的---->nms
        """
        return (num of left proposal(around 2000),*5)
        blob[:,0]==0
        blob[:,1:5]: x1,y1,x2,y2(in the original image)
        """

        #'gt_boxes': [[11,22,33,44,16],[22,33,44,55,8]]boxes +classes;
        (self.feed('rpn_rois','gt_boxes')
             .proposal_target_layer(n_classes,name = 'roi-data'))
        #choose pred_box by itself and the overlaps with the gt_boxes
        """
(1) rois: (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
          [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
(2) the final classes of the ground truth correspounding to per pred box [num of finally left proposal,1] 
        for ex: [[9],[15],[15],[15],[9],[9]....]
(3): num of finally left proposals * 4*21: [dx,dy,dw,dh](relative to gt) of 1 class in 21
(4): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
(5): num of finally left proposals * 4*21: [true,true,true,true] of 1 class in 21; [false, false, false, false] in left of the classes
"""


        #========= RCNN ============
        (self.feed('conv5_3', 'roi-data')
             .roi_pool(7, 7, 1.0/16, name='pool_5')
         # 主要用roi_data[0]
         # output: (num_rois, h, w, channels)  around(42, 7, 7, 512)
             .fc(4096, name='fc6')
         # num of left proposals * 4096
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
         # num of left proposals * 4096
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
         # num of left proposals * 21
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))

