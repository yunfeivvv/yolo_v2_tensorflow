import tensorflow as tf
import utils.augmentation as aug

def _cal_iou_wh(h1, w1, h2, w2):
    intersect_w = tf.cond(tf.less(w1, w2), lambda: w1, lambda: w2)
    intersect_h = tf.cond(tf.less(h1, h2), lambda: h1, lambda: h2)
    intersect = intersect_h * intersect_w
    union = h1 * w1 + h2 * w2 - intersect
    iou = tf.clip_by_value(intersect / union, 0.0, 1.0)
    return iou


def _pad_bboxes(bboxes, max_num):
    num_bboxes = tf.shape(bboxes)[0]
    cond = tf.less_equal(num_bboxes, max_num)
    padded_bboxes = tf.cond(cond,
                            lambda:tf.pad(bboxes, [[0, max_num - num_bboxes], [0, 0]]),
                            lambda:tf.slice(bboxes, [0, 0], [max_num, 4]))
    padded_bboxes = tf.reshape(padded_bboxes,[max_num, 4])
    return padded_bboxes

def _preprocessed_bboxes(raw_clses, raw_bboxes, **kwargs):
    basic_shape = [kwargs['grid_size'], kwargs['grid_size'], kwargs['num_anchors']]
    gs_f = tf.to_float(kwargs['grid_size'])
    nc = kwargs['num_classes']

    with tf.variable_scope('preprocessed_label'):

        def condition(i, raw_clses, raw_bboxes, processed_conf, processed_bboxes, processed_clses):
            r = tf.less(i, tf.shape(raw_clses)[0])
            return r

        def body(i, raw_clses, raw_bboxes, processed_conf, processed_bboxes, processed_clses):
            raw_cls = raw_clses[i]
            raw_bbox = raw_bboxes[i]
            raw_bbox = raw_bbox * gs_f

            ymin, xmin, ymax, xmax = tf.unstack(raw_bbox, axis=-1)
            ymid = (ymin + ymax) * 0.5
            xmid = (xmin + xmax) * 0.5
            height = ymax - ymin
            width = xmax - xmin

            # for bbox
            bbox_tmp = tf.stack([ymid, xmid, height, width], axis=-1) / gs_f
            bbox_tmp = tf.reshape(bbox_tmp, [1, 1, 1, 4])
            bbox_tmp = tf.tile(bbox_tmp, basic_shape + [1])
            # for conf
            conf_tmp = tf.ones_like(processed_conf)
            # for cls
            raw_cls = tf.reshape(raw_cls, [1, 1, 1]) - kwargs['cls_offset']
            one_hot = tf.one_hot(raw_cls, nc, 1., 0., dtype=tf.float32)
            one_hot = tf.tile(one_hot, basic_shape + [1])

            # 分配anchor
            best_anchor = -1
            max_iou = -1.
            for j in range(kwargs['num_anchors']):
                anchor = kwargs['anchors'][j]
                iou = _cal_iou_wh(height, width, tf.to_float(anchor[0]), tf.to_float(anchor[1]))
                [best_anchor, max_iou] = tf.cond(tf.less(max_iou, iou),
                                                 lambda: [j, iou],
                                                 lambda: [best_anchor, max_iou])
            # 制作mask
            mask = tf.SparseTensor(indices=[[tf.to_int64(ymid), tf.to_int64(xmid), tf.to_int64(best_anchor), 0]],
                                   values=[True], dense_shape=basic_shape + [1])
            mask = tf.sparse_tensor_to_dense(mask, default_value=False)

            # 更新
            processed_conf   = tf.where(mask                        , conf_tmp, processed_conf)
            processed_bboxes = tf.where(tf.tile(mask, [1, 1, 1, 4 ]), bbox_tmp, processed_bboxes)  # norm!!!
            processed_clses  = tf.where(tf.tile(mask, [1, 1, 1, nc]), one_hot , processed_clses)

            return [i + 1, raw_clses, raw_bboxes, processed_conf, processed_bboxes, processed_clses]

        i = 0
        processed_conf   = tf.zeros(basic_shape + [1])
        processed_bboxes = tf.zeros(basic_shape + [4])
        processed_clses  = tf.zeros(basic_shape + [nc])

        [i, raw_clses, raw_bboxes, processed_conf, processed_bboxes, processed_clses] = \
            tf.while_loop(condition, body, [i, raw_clses, raw_bboxes, processed_conf, processed_bboxes, processed_clses])

        processed_label = tf.concat([processed_conf, processed_bboxes, processed_clses], axis=-1)
        padded_bboxes = _pad_bboxes(raw_bboxes, kwargs['max_bboxes_per_img']) # 可以移到外面

        return processed_label, padded_bboxes


def _processed_image(raw_img, raw_bboxes, **kwargs):
    with tf.variable_scope('preprocessed_image'):

        if raw_img.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        if not raw_img.dtype == tf.float32:
            raw_img = tf.to_float(raw_img)
            raw_img = raw_img / 255.

        processed_img = tf.image.resize_images(raw_img, [kwargs['img_size'], kwargs['img_size']],
                                             method=tf.image.ResizeMethod.BILINEAR,
                                             align_corners=False)
        # augmentation
        processed_img, raw_bboxes = aug.random_flip_left_right(processed_img, raw_bboxes)
        # todo
        # begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(image),
        #                                                                     bounding_boxes=tf.expand_dims(gbboxes, 0),
        #                                                                     min_object_covered=0.3,
        #                                                                     aspect_ratio_range=(0.9, 1.1),
        #                                                                     area_range=(0.1, 1.0),
        #                                                                     max_attempts=200,
        #                                                                     use_image_if_no_bounding_boxes=True)
        #
        # image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox_for_draw)
        # distorted_image = tf.slice(image, begin, size)
        processed_img = aug.distort_color(processed_img)
        processed_img = processed_img * 255.

        return processed_img, raw_bboxes


def preprocess_for_train(raw_img, raw_clses, raw_bboxes, **kwargs):
    # processed image
    processed_img, raw_bboxes = _processed_image(raw_img, raw_bboxes, **kwargs)

    # processed clses and bboxes
    processed_label, padded_bboxes = _preprocessed_bboxes(raw_clses, raw_bboxes, **kwargs)

    return processed_img, processed_label, padded_bboxes





# def _yxyx2yxhw(yxyx):
#     ymin, xmin, ymax, xmax = tf.unstack(yxyx, axis=-1)
#     y = (ymin + ymax) * 0.5
#     x = (xmin + xmax) * 0.5
#     h = ymax - ymin
#     w = xmax - xmin
#     yxhw = tf.stack([y,x,h,w], axis=-1)
#     return yxhw

# def _preprocessed_bboxes_v2(raw_clses, raw_bboxes, **kwargs):
#     '''
#     params:
#         - raw_clses:    [num_gt_bboxes, 1], 1~20(不包括背景)
#         - raw_bboxes:   [num_gt_bboxes, 4], yxyx
#         - kwargs:
#             NET_cfg
#     return:
#
#     '''
#     gs = kwargs['grid_size']
#     gs_f = tf.to_float(kwargs['grid_size'])
#     na = kwargs['num_anchors']
#     nc = kwargs['num_classes']
#
#     anchors = tf.reshape(kwargs['anchors'], [na, 1, 2])
#
#     true_yxhw = _yxyx2yxhw(raw_bboxes) * gs_f
#     true_yx = tf.to_int64(true_yxhw[..., 0:2])
#     true_hw = true_yxhw[..., 2:4]
#     true_hw = tf.reshape(true_hw, [1, -1, 2])
#
#     inter_hw = tf.minimum(true_hw, anchors)
#     inter_square = inter_hw[..., 0] * inter_hw[..., 1]
#     square1 = true_hw[..., 0] * true_hw[..., 1]
#     square2 = anchors[..., 0] * anchors[..., 1]
#     iou_scores = inter_square / (square1 + square2 - inter_square)      # na x num_gt_bboxes
#     best_anchor = tf.expand_dims(tf.argmax(iou_scores, axis=-2), -1)    # 每个gt对应的anchor， num_gt_bboxes
#
#     raw_clses = raw_clses - kwargs['cls_offset']
#     one_hot_clses = tf.one_hot(raw_clses, nc, 1., 0., dtype=tf.float32)
#
#     indices = tf.concat([true_yx, best_anchor, tf.zeros_like(best_anchor,tf.int64)], -1)
#     conf = tf.sparse_tensor_to_dense(tf.SparseTensor(indices, [1.0], [gs, gs, na, 1]), 0.0)
#
#     bbox = true_yxhw
#     bbox = tf.where(conf, true_yxhw[best_anchor], tf.zeros_like)