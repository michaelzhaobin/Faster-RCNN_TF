# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    # scales = [8, 16, 32]; no other 
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    """
  [[-1.5, 3 ,19.5, 14]
   [1, 0.5, 16, 16.5]
   [3.5 -1.5 13.5 19.5]
  ]
  """
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
#      [[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])
    return anchors

def _whctrs(anchor):
  # anchor: [1,1,16,16]
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    #16
    h = anchor[3] - anchor[1] + 1
    #16
    x_ctr = anchor[0] + 0.5 * (w - 1)
    #8.5
    y_ctr = anchor[1] + 0.5 * (h - 1)
    #8.5
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
  #[23,16,11] [11,16,22] 8.5 8.5
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    """
    [[23],
    [16],
    [11]
    ]
    """
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors
  """
  [[-1.5, 3 ,19.5, 14]
   [1, 0.5, 16, 16.5]
   [3.5 -1.5 13.5 19.5]
  ]
  """

def _ratio_enum(anchor, ratios):
  #anchor: [1,1,16,16] (not dimention); ratios=[0.5, 1, 2]
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # 16 16 8.5 8.5
    size = w * h
    #256
    size_ratios = size / ratios
    #[512 256 128]
    ws = np.round(np.sqrt(size_ratios))
    #[23 16 11]a
    hs = np.round(ws * ratios)
    #[11, 16, 22]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
  """
  [[-1.5, 3 ,19.5, 14]
   [1, 0.5, 16, 16.5]
   [3.5 -1.5 13.5 19.5]
  ]
  """

def _scale_enum(anchor, scales):
  #anchor: [-1.5, 3 ,19.5, 14] scales: [8, 16, 32]
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
