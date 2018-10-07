'''backbone

输入：256x256x3

| --- feats size --- | --- num kernels --- | --- shape kernel --- | --- stride --- | --- name --- |

    256                        32                   3                    1              conv1
    128                        64                   3                    2              downsample1
    *** block1 ***
    128                        32                   1                    1              block1/conv1(shrink)
    128                        64                   3                    1              block1/conv2(expand)
     -                         --                   -                    -              block1/residual
    64                         128                  3                    2              downsample2
    *** block2 ***
    64                         64                   1                    1              block2/conv1
    64                         128                  3                    1              block2/conv2
    -                          --                   -                    -              block2/residual
    *** block3 *** (和2相同)
    32                         256                  3                    2              downsample3
    *** block4 ***
    32                         128                  1                    1              block4/conv1
    32                         256                  3                    1              block4/conv2
    -                          --                   -                    -              block4/residual
    *** block5 *** (和4相同)
    *** block6 *** (和4相同)
    *** block7 *** (和4相同)
    *** block8 *** (和4相同)
    *** block9 *** (和4相同)
    *** block10 *** (和4相同)
    *** block11 *** (和4相同)
    16                         512                  3                    2              downsample4
    *** block12 ***
    16                         256                  1                    1              block12/conv1
    16                         512                  3                    1              block12/conv2
    -                          --                   -                    -              block12/residual
    *** block13 *** (和12相同)
    *** block14 *** (和12相同)
    *** block15 *** (和12相同)
    *** block16 *** (和12相同)
    *** block17 *** (和12相同)
    *** block18 *** (和12相同)
    *** block19 *** (和12相同)
    8                          1024                 3                   2               downsample5
    *** block20 ***
    8                          512                  1                    1              block20/conv1
    8                          1024                 3                    1              block20/conv2
    -                          --                   -                    -              block20/residual
    *** block21 *** (和20相同)
    *** block22 *** (和20相同)
    *** block23 *** (和20相同)

    Average pooling

    1                          1000                 1                    1              connected

    Softmax

'''
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


#
#   utils.py
#
def cal_iou(boxes1, boxes2, type1='yxhw', type2='yxhw'):

    #to (y1, x1, y2, x2)
    boxes1_t = tf.cond(tf.cast(type1=='yxyx', tf.bool),
                       lambda: boxes1,
                       lambda: tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                         boxes1[..., 1] + boxes1[..., 3] / 2.0], axis=-1))
    boxes2_t = tf.cond(tf.cast(type2=='yxyx', tf.bool),
                       lambda: boxes2,
                       lambda: tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                         boxes2[..., 1] + boxes2[..., 3] / 2.0], axis=-1))

    lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

    # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    # calculate the box1 square and boxs2 square
    square1 = boxes1[..., 2] * boxes1[..., 3]  # h*w
    square2 = boxes2[..., 2] * boxes2[..., 3]  # h*w

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


#
#   network.py
#
def v3_arg_scope(is_training):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.leaky_relu,
                        normalizer_fn=slim.batch_norm,
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(1.0)),\
        slim.arg_scope([slim.batch_norm],
                       is_training=is_training) as sc:
        return sc

def block(inputs, num, scope):
    with tf.variable_scope(scope, values=[inputs]):
        net = slim.conv2d(inputs, num//2, 1, scope='conv1')
        net = slim.conv2d(net, num, 3, scope='conv2')
        net = inputs + net
        return net

def upsample(net, out_shape):
    '''
    8*8 -> 10*10 -> 20*20 -> 16*16
       (pad)  (resize)  (crop)
    '''
    with tf.variable_scope('upsample', values=[net]):
        net = tf.pad(net, [[0, 0], [1, 1],[1, 1], [0, 0]], mode='SYMMETRIC')
        h = out_shape[0] + 4
        w = out_shape[1] + 4
        net = tf.image.resize_bilinear(net, (h, w))
        net = net[:, 2:-2, 2:-2, :]
        return net

def darknet_53_backbone(inputs):

    with tf.variable_scope('darknet_53', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        count = 0
        with slim.arg_scope([slim.conv2d], outputs_collections=end_points_collection):

            net = slim.conv2d(inputs, 32, 3, scope='conv1')
            net = slim.conv2d(net, 64, 3, 2, scope='downsample1')

            for _ in range(1):
                count += 1
                net = block(net, 64, 'block%s'%(count))

            net = slim.conv2d(net, 128, 3, 2, scope='downsample2')

            for _ in range(2):
                count += 1
                net = block(net, 128, 'block%s'%(count))

            net = slim.conv2d(net, 256, 3, 2, scope='downsample3')

            for _ in range(8):
                count += 1
                net = block(net, 256, 'block%s'%(count))

            net = slim.conv2d(net, 512, 3, 2, scope='downsample4')

            for _ in range(8):
                count += 1
                net = block(net, 512, 'block%s'%(count))

            net = slim.conv2d(net, 1024, 3, 2, scope='downsample5')

            for _ in range(4):
                count += 1
                net = block(net, 1024, 'block%s'%(count))

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    pred1_feats = end_points['darknet_53/block23/conv2']    # stride 32, channels 1024, size 8
    pred2_feats = end_points['darknet_53/block19/conv2']    # stride 16, channels 512 , size 16
    pred3_feats = end_points['darknet_53/block11/conv2']    # stride 8 , channels 256 , size 32

    return pred1_feats, pred2_feats, pred3_feats

def v3_head(feats1, feats2, feats3, num_classes=20):

    with tf.variable_scope('header', values=[feats1, feats2, feats3]):
        net = feats1
        net = slim.conv2d(net, 512 , 1, 1, scope='conv1')
        net = slim.conv2d(net, 1024, 3, 1, scope='conv2')
        net = slim.conv2d(net, 512 , 1, 1, scope='conv3')
        net = slim.conv2d(net, 1024, 3, 1, scope='conv4')
        net = slim.conv2d(net, 512 , 1, 1, scope='conv5')

        with tf.variable_scope('pred1', values=[net]):
            pred1 = slim.conv2d(net, 1024, 3, 1, scope='conv1')
            pred1 = slim.conv2d(pred1, 3*(5+num_classes), 1, 1,
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='conv2')

        net = slim.conv2d(net, 256 , 1, 1, scope='conv6')
        net = tf.concat([feats2, upsample(net, feats2.shape.as_list()[1:3])], axis=-1)  # yolo v3: concat， fpn: add

        with tf.variable_scope('pred2', values=[net]):
            pred2 = slim.conv2d(net, 512, 3, 1, scope='conv1')
            pred2 = slim.conv2d(pred2, 3*(5+num_classes), 1, 1,
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='conv2')

        net = slim.conv2d(net, 128 , 1, 1, scope='conv7')
        net = tf.concat([feats3, upsample(net, feats3.shape.as_list()[1:3])], axis=-1)

        with tf.variable_scope('pred3', values=[net]):
            pred3 = slim.conv2d(net, 256, 3, 1, scope='conv1')
            pred3 = slim.conv2d(pred3, 3*(5+num_classes), 1, 1,
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='conv2')

        return pred1, pred2, pred3

def v3_norm(inputs):
    if not inputs.dtype == tf.float32:
        inputs = tf.to_float(inputs)
    inputs = inputs / 255.
    return inputs
#
#   loss.py
#
def loss_layer(pred, gt, anchor):
    '''
    params:
        - pred  :
        - gt    : feature map 尺寸作为单位长度。
        - anchor: feature map 尺寸作为单位长度。
    return:
        - loss  : 当前尺寸产生的loss
    '''
    na = 3
    nc = 20
    gs_h, gs_w = gt.shape.as_list()[1:3]
    lambda_coord = 1.
    lambda_class = 1.
    lambda_pos = 1.
    lambda_neg = 0.5

    with tf.variable_scope('prepare', values=[pred, gt, anchor]):
        # prediction
        pred = tf.reshape(pred, [-1, gs_h, gs_w, na, 5 + nc])
        pred_conf = pred[:, :, :, :, 0:1]
        pred_yx   = pred[:, :, :, :, 1:3]
        pred_hw   = pred[:, :, :, :, 3:5]
        pred_cls  = pred[:, :, :, :, 5:5 + nc]
        # ground truth
        gt_conf = gt[:, :, :, :, 0:1]
        gt_yx   = gt[:, :, :, :, 1:3]
        gt_hw   = gt[:, :, :, :, 3:5]
        gt_cls  = gt[:, :, :, :, 5:5 + nc]
        # prior
        p_x, p_y = tf.meshgrid([i for i in range(gs_h)],  # height
                               [i for i in range(gs_w)])  # width
        p_x  = tf.cast(tf.reshape(p_x, [1, gs_h, gs_w, 1, 1]), tf.float32)
        p_y  = tf.cast(tf.reshape(p_y, [1, gs_h, gs_w, 1, 1]), tf.float32)
        p_yx = tf.concat([p_y, p_x], axis=4)
        p_hw = tf.reshape(anchor, [1, 1, 1, na, 2])
        p_bbox = tf.concat([tf.tile(p_yx, [1, 1, 1, 3, 1])+0.5,
                            tf.tile(p_hw, [1, gs_h, gs_w, 1, 1])], axis=-1) # 1x13x13x3x4

    with tf.variable_scope('pred_to_hat', values=[pred_yx, pred_hw, p_yx, p_hw]):
        hat_yx   = tf.sigmoid(pred_yx) + p_yx
        hat_hw   = tf.exp(pred_hw) * p_hw
        # hat_conf = tf.sigmoid(pred_conf)
        hat_conf = pred_conf
        hat_cls  = pred_cls

    with tf.variable_scope('generate_mask'):  # todo 这里可能有问题
        ious = cal_iou(tf.expand_dims(p_bbox, axis=-2),         # 1,  h, w, 3, 1, 4
                       tf.expand_dims(gt[..., 1:5], axis=-3))   # 16, h, w, 1, 3, 4 / r: 16, h, w, 3, 3
        best_ious = tf.reduce_max(ious, axis=-1)                # 16, h, w, 3
        # 对于每个default，所有与他的iou小于threshold，那么Negative。
        neg_mask = tf.expand_dims(tf.to_float(best_ious<0.5), axis=-1) * (1. - gt_conf)
        pos_mask = gt_conf

    with tf.variable_scope('loss'):
        num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.)
        num_neg = tf.maximum(tf.reduce_sum(neg_mask), 1.)

        loss_hw = tf.reduce_sum(tf.square(hat_hw-gt_hw)*pos_mask) / num_pos
        loss_yx = tf.reduce_sum(tf.square(hat_yx-gt_yx)*pos_mask) / num_pos

        loss_cls = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cls , logits=hat_cls) * pos_mask) / num_pos
        loss_pos = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf, logits=hat_conf) * pos_mask) / num_pos
        loss_neg = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf, logits=hat_conf) * neg_mask) / num_neg

        total_loss = loss_hw * lambda_coord + loss_yx * lambda_coord + loss_cls * lambda_class + loss_pos * lambda_pos + loss_neg * lambda_neg

    return total_loss

def _yxyx2yxhw(yxyx):
    ymin, xmin, ymax, xmax = tf.unstack(yxyx, axis=-1)
    y = (ymin + ymax) * 0.5
    x = (xmin + xmax) * 0.5
    h = ymax - ymin
    w = xmax - xmin
    yxhw = tf.stack([y,x,h,w], axis=-1)
    return yxhw

def loss_fn(preds, gts):
    anchors = [[[10/416 , 13/416], [16/416 , 30/416 ], [33/416 , 23/416 ],],  # coco
               [[30/416 , 61/416], [62/416 , 45/416 ], [59/416 , 119/416],],
               [[116/416, 90/416], [156/416, 198/416], [373/416, 326/416]]]
    anchors = anchors[::-1]
    loss = 0.
    for p, t, a in zip(preds, gts, anchors):
        loss += loss_layer(p, t, a)
    return loss


#
#   processing.py
#
# tip1:
#       生成positive mask：遍历每个gt，最大iou对应的anchor设定为1；
#       生成negative mask：遍历每个anchor，和所有的gt之间的iou都小于threshold，那么设定为1；(放到计算loss的函数中)
#       两种mask生成的途径不同，因此要单独生成。
#       （或许可以同时生成，暂时没想出解决方案）
def _cal_iou_wh(h1, w1, h2, w2):
    intersect_w = tf.cond(tf.less(w1, w2), lambda: w1, lambda: w2)
    intersect_h = tf.cond(tf.less(h1, h2), lambda: h1, lambda: h2)
    intersect = intersect_h * intersect_w
    union = tf.maximum(h1 * w1 + h2 * w2 - intersect, 1e-8)
    iou = intersect / union
    return iou

def _pad_label(gt, max_num):
    num_obj = tf.shape(gt)[0]
    padded_true = tf.cond(tf.less_equal(num_obj, max_num),
                          lambda:tf.pad(gt, [[0, max_num - num_obj], [0, 0]]),
                          lambda:tf.slice(gt, [0, 0], [max_num, 5]))
    padded_true = tf.reshape(padded_true,[max_num, 5])
    return padded_true

def gen_pos(gt, idx, cls, ymid, xmid, height, width):
    '''
        * gt:   tensor, [grid_h, grid_w, 3, 25]
                一个空的框架，函数的目的就是往这个框架的合适的位置填充合适的值
        * idx:  scale, int64, 0/1/2, anchor with max iou
                决定往gt的第3维的那个位置填充数值。
        * cls:  scale, int64, 0~nc
                转化成oe-hot形式，填充类别信息
        * ymid/xmid:    scale, float32,
                决定往gt的第1和第2维的那个位置填充数值
        * height/width: scale, float32,
                待填充的coord信息
    '''
    gs_h, gs_w, n, nc = gt.shape.as_list()
    nc = nc - 5
    basic_shape = [gs_h, gs_w, n]
    cls_offset = 1

    # for bbox
    bbox = tf.stack([ymid, xmid, height, width], axis=-1)
    bbox = tf.reshape(bbox, [1, 1, 1, 4])
    bbox = tf.tile(bbox, basic_shape + [1])
    # for conf
    conf = tf.ones(basic_shape + [1])
    # for cls
    cls = tf.reshape(cls, [1, 1, 1]) - cls_offset
    one_hot = tf.one_hot(cls, nc, 1., 0., dtype=tf.float32)
    one_hot = tf.tile(one_hot, basic_shape + [1])

    mask = tf.SparseTensor(indices=[[tf.to_int64(ymid * gs_h), tf.to_int64(xmid * gs_w), tf.to_int64(idx), 0]],
                           values=[True], dense_shape=[gs_h, gs_w, n, 1])
    mask = tf.sparse_tensor_to_dense(mask, default_value=False)

    # 更新
    processed_conf = tf.where(mask,                         conf,   gt[..., 0:1])
    processed_bbox = tf.where(tf.tile(mask, [1, 1, 1, 4]),  bbox,   gt[..., 1:5])
    processed_cls  = tf.where(tf.tile(mask, [1, 1, 1, nc]), one_hot, gt[..., 5:])

    r = tf.concat([processed_conf, processed_bbox, processed_cls], axis=-1)
    return r

def preprocess_label(raw_clses, raw_bboxes):
    # img.shape = 416. 416. 3
    '''
    params:
    * raw_clses:
    * raw_bboxes:

    return:
    * gt_1: tf.float32, tensor, 13x13x3x(5+nc)
            大目标会放在这里面
    * gt_2:
    * gt_3:
    '''
    nc = 20
    na = 9
    gs_h_1, gs_w_1 = 13, 13
    gs_h_2, gs_w_2 = 26, 26
    gs_h_3, gs_w_3 = 52, 52
    img_h , img_w  = 416., 416.
    anchors = [[10 , 13], [16 , 30 ], [33 , 23 ],  # coco
               [30 , 61], [62 , 45 ], [59 , 119],
               [116, 90], [156, 198], [373, 326]]

    def cond(i, raw_clses, raw_bboxes, gt_1, gt_2, gt_3):
        r = tf.less(i, tf.shape(raw_clses)[0])
        return r

    def body(i, raw_clses, raw_bboxes, gt_1, gt_2, gt_3):
        '''
        首先根据每个anchor，然后找到grid。
        '''

        raw_cls = raw_clses[i]
        raw_bbox = raw_bboxes[i]
        ymid, xmid, height, width = tf.unstack(_yxyx2yxhw(raw_bbox), axis=-1)

        pos_idx = -1
        max_iou = -1.
        for j in range(na):
            anchor = anchors[j]
            iou = _cal_iou_wh(height, width, tf.to_float(anchor[0])/img_h, tf.to_float(anchor[1])/img_w)
            [pos_idx, max_iou] = tf.cond(tf.less(max_iou, iou),
                                          lambda: [j, iou],
                                          lambda: [pos_idx, max_iou])

        pred_fn_pairs = {
            tf.equal(pos_idx//3, 0): lambda: [gt_1, gt_2, gen_pos(gt_3, pos_idx%3, raw_cls, ymid, xmid, height, width)],
            tf.equal(pos_idx//3, 1): lambda: [gt_1, gen_pos(gt_2, pos_idx%3, raw_cls, ymid, xmid, height, width), gt_3],
            tf.equal(pos_idx//3, 2): lambda: [gen_pos(gt_1, pos_idx%3, raw_cls, ymid, xmid, height, width), gt_2, gt_3],
        }

        gt_1, gt_2, gt_3 = tf.case(pred_fn_pairs)

        return [i+1, raw_clses, raw_bboxes, gt_1, gt_2, gt_3]

    i = 0
    gt_1 = tf.zeros([gs_h_1, gs_w_1, na//3, 5+nc])
    gt_2 = tf.zeros([gs_h_2, gs_w_2, na//3, 5+nc])
    gt_3 = tf.zeros([gs_h_3, gs_w_3, na//3, 5+nc])

    [i, raw_clses, raw_bboxes, gt_1, gt_2, gt_3] = \
        tf.while_loop(cond, body, [i, raw_clses, raw_bboxes, gt_1, gt_2, gt_3])

    return gt_1, gt_2, gt_3

def preprocess_image(img, bboxes):

    img_h = 416
    img_w = 416
    if img.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if not img.dtype == tf.float32:
        img = tf.to_float(img)
        img = img / 255.

    img = tf.image.resize_images(img, [img_h, img_w],
                                 method=tf.image.ResizeMethod.BILINEAR,
                                 align_corners=False)
    # todo add augmentation
    img = img * 255.

    return img, bboxes

def preprocess_for_train(img, raw_clses, raw_bboxes, **kwargs):

    max_obj_per_img = 30    # todo

    img, raw_bboxes = preprocess_image(img, raw_bboxes)

    gt_1, gt_2, gt_3 = preprocess_label(raw_clses, raw_bboxes)

    bboxes_pad = _pad_label(raw_bboxes, max_obj_per_img)

    return img, gt_1, gt_2, gt_3, bboxes_pad


if __name__ == '__main__':
    #
    #   network.py
    #
    # x = tf.placeholder(tf.float32, [10, 256, 256, 3])
    # with slim.arg_scope(v3_arg_scope()):
    #     f1, f2, f3 = darknet_53_backbone(x, True)
    #     y1, y2, y3 = v3_head(f1, f2, f3, 20)
    #
    # writer = tf.summary.FileWriter('./log', tf.get_default_graph())


    #
    #   processing.py 测试
    #
    # raw_cls = tf.constant([[10], [4], [6]], tf.int64)
    # raw_bbox = tf.constant([[0.1, 0.1, 0.1+16/416, 0.1+30/416],                 # 小目标，应该分到gt3
    #                         [0.2, 0.2, 0.2+62/416, 0.2+45/416],                 # 中目标，应该分到gt2
    #                         [0.4, 0.1, 0.4+156/416, 0.1+198/416]], tf.float32)  # 大目标，应该分到gt1
    # g1, g2, g3 = preprocess_label(raw_cls, raw_bbox)
    # with tf.Session() as sess:
    #     gg1, gg2, gg3 = sess.run([g1, g2, g3])
    # print(gg1[np.where(gg1>0)], '\n\n',
    #       gg2[np.where(gg2>0)], '\n\n',
    #       gg3[np.where(gg3>0)])

    #
    #   loss.py
    #
    p1 = tf.random_uniform((16, 13, 13, 3, 25))
    p2 = tf.random_uniform((16, 26, 26, 3, 25))
    p3 = tf.random_uniform((16, 52, 52, 3, 25))
    t1 = tf.random_uniform((16, 13, 13, 3, 25))
    t2 = tf.random_uniform((16, 26, 26, 3, 25))
    t3 = tf.random_uniform((16, 52, 52, 3, 25))
    loss = loss_fn([p1, p2, p3], [t1, t2, t3])

    with tf.Session() as sess:
        print(sess.run(loss))
