import numpy as np
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py



DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            #data_dict.keys: ['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2', 
            #'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key
                            if not ignore_missing:

                                raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        # input is placeholder defined by tf: [None, None, None, 3]; c_i = 3
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                conv = convolve(input, kernel)
                # conv = tf.nn.conv2d(input, tf.get_variable('weights', [k_h, k_w, c_i/group, c_o], initializer = 
                # tf.truncated_normal_initializer(0.0, stddev=0.01), trainable = True), [1, s_h, s_w, 1], padding=padding)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        """
        input:'conv5_3', 'roi-data'
        roi-data:(1) the final classes of the ground truth correspounding to per pred box [num of finally left proposal,1] 
                      for ex: [[9],[15],[15],[15],[9],[9]....]
                 (2) (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
                      [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
                 (3): num of finally left proposals * 4*21: [dx,dy,dw,dh] of 1 class in 21
                 (4): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
                 (5): num of finally left proposals * 4*21: [true,true,true,true] of 1 class in 21; [false, false, false, false] in left of the classes
        roi_pool(7, 7, 1.0/16, name='pool_5')
        """
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    #input: 'rpn_cls_prob_reshape'(1,14,14,18),'rpn_bbox_pred'(1,14,14,36)((dx,dx,dw,dh)),'im_info': [[max_length, max_width, im_scale]]
    # _feat_stride:[16,];  anchor_scales:[8,16,32]; cfg_key = 'TRAIN'
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32]),[-1,5],name =name)
    """
    return (num of left proposal,*5)
    blob[:,0]==0
    blob[:,1:5]: x1,y1,x2,y2
    """

    @layer
    # input: [1,14,14,18]('rpn_cls_score'); [[11,22,33,44,0],[22,33,44,55,2]]boxes +classes('gt_boxes'); 
    #        [[max_length, max_width, im_scale]]('im_info'); [1,maxL,maxH,3]('data')
    # _feat_stride = [16,]; anchor_scales = [8, 16, 32]
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:

            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    @layer
    #input: rpn_rois: (num of left proposal,*5) blob[:,0]==0 blob[:,1:5]: x1,y1,x2,y2
    #       gt_boxes: [[11,22,33,44,0],[22,33,44,55,2]]boxes +classes
    #classes: 21
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:

            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(proposal_target_layer_py,[input[0],input[1],classes],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
            """
            (1) the final classes of the ground truth correspounding to per pred box [num of finally left proposal,1] 
                  for ex: [[9],[15],[15],[15],[9],[9]....]
            (2) (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
                  [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
            (3): num of finally left proposals * 4*21: [dx,dy,dw,dh] of 1 class in 21
            (4): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
            (5): num of finally left proposals * 4*21: [true,true,true,true] of 1 class in 21; [false, false, false, false] in left of the classes
            """
            rois = tf.reshape(rois,[-1,5] , name = 'rois') 
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

           
            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
            """
(1) rois: (num of finally left proposal, 5) blob[:,0]=0; blob[:-2,1:5] = x1,y1,x2,y2(pred box); blob[-2:,1:5] = x1,y1,x2,y2(gt_box)
          [0:fg_rois_per_this_image]: the left foregound; [fg_rois_per_this_image:]:the left background
(2) the final classes of the ground truth correspounding to per pred box [num of finally left proposal,1] 
        for ex: [[9],[15],[15],[15],[9],[9]....]

(3): num of finally left proposals * 4*21: [dx,dy,dw,dh] of 1 class in 21
(4): num of finally left proposals * 4*21: [1, 1, 1, 1] of 1 class
(5): num of finally left proposals * 4*21: [true,true,true,true] of 1 class in 21; [false, false, false, false] in left of the classes
"""


    @layer
    def reshape_layer(self, input, d,name):
        #input: [1,14,14,18]
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
        # d = 18; input(1,14*9,14,2)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
        # tf.transpose(tf.reshape(in(1,2,14*9,14),[1,18,14*9/18*2,14]),[0,2,3,1])
        # output: [1, 14,14,18]
        else:
        """
        rpn_cls_score_reshape: input:[1,14,14,18]
        tf.transpose(tf.reshape(in(1,18,14,14),[1,2,14*18/2,14]),[0,2,3,1]) --------(1,14*9,14,2)
        """
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input,
                              scales_base,
                              num_scale_base,
                              num_per_octave,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)
    

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            # input: (1,126(9*14),14,2)
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits))(all sum)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)
