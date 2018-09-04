import tensorflow as tf
import utils.augmentation as aug


def _cal_iou_wh(h1, w1, h2, w2):

    intersect_w = tf.cond(tf.less(w1, w2), lambda:w1, lambda:w2)
    intersect_h = tf.cond(tf.less(h1, h2), lambda:h1, lambda:h2)
    intersect = intersect_h * intersect_w

    union = h1*w1 + h2*w2 - intersect

    iou = tf.clip_by_value(intersect/union, 0.0, 1.0)
    return iou

# raw(sparse) to gt(dense) todo 理清和loss的关系
'''

'''
def _preprocess_boxes(raw_clses, raw_bboxes, **kwargs):

    basic_shape = [kwargs['grid_size'], kwargs['grid_size'], kwargs['num_anchors']]
    with tf.variable_scope('preprocess_label'):

        def condition(i, raw_clses, raw_bboxes, score, process_bboxes, process_clses):
            r = tf.less(i, tf.shape(raw_clses)[0])
            return r

        def body(i, raw_clses, raw_bboxes, score, process_bboxes, process_clses):

            raw_cls = raw_clses[i]
            raw_bbox = raw_bboxes[i]
            raw_bbox = raw_bbox * kwargs['grid_size']

            # 按照中点分配mask，根据default bboxes与gt bbox之间的iou，如果两者之间的iou大于阈值(指定)，那么分配mask。
            ymin, xmin, ymax, xmax = tf.unstack(raw_bbox)
            ymid = (ymin + ymax) * 0.5
            xmid = (xmin + xmax) * 0.5
            height = ymax - ymin
            width = xmax - xmin

            # TODO:bbox精确到哪个anchor，score以及raw_cls需不需要精确到哪个anchor?
            # for bbox
            bbox_tmp = tf.stack([ymid, xmid, height, width], axis=-1)
            bbox_tmp = tf.reshape(bbox_tmp, [1, 1, 1, 4])
            bbox_tmp = tf.tile(bbox_tmp, basic_shape + [1])
            # for score
            score_tmp = tf.ones_like(score)
            # for raw_cls
            raw_cls = tf.reshape(raw_cls, [1, 1, 1]) - kwargs['cls_offset']
            one_hot = tf.one_hot(raw_cls, 20, 1., 0., dtype=tf.float32)
            one_hot = tf.tile(one_hot, basic_shape + [1])

            # 分配anchor
            best_anchor = -1
            max_iou = -1.
            for j in range(kwargs['num_anchors']):
                anchor = kwargs['anchors'][j]
                iou = _cal_iou_wh(height, width, anchor[0], anchor[1])
                [best_anchor, max_iou] = tf.cond(tf.less(max_iou, iou),
                                                 lambda:[j, iou],
                                                 lambda:[best_anchor, max_iou])
            # 制作mask
            yidx = tf.to_int64(ymid - kwargs['grid_size']/kwargs['img_size'])
            xidx = tf.to_int64(xmid - kwargs['grid_size']/kwargs['img_size'])
            best_anchor = tf.to_int64(best_anchor)
            mask = tf.SparseTensor(indices=[[yidx, xidx, best_anchor, 0]], values=[True],
                                   dense_shape=basic_shape + [1])
            mask = tf.sparse_tensor_to_dense(mask, default_value=False)

            # 根据mask分配标签
            score = tf.where(mask, score_tmp, score)
            process_bboxes = tf.where(tf.tile(mask, [1, 1, 1, 4]), bbox_tmp, process_bboxes)
            process_clses= tf.where(tf.tile(mask, [1, 1, 1, 20]), one_hot, process_clses)

            return [i+1, raw_clses, raw_bboxes, score, process_bboxes, process_clses]

        i = 0
        score = tf.zeros(basic_shape + [1])
        process_bboxes = tf.zeros(basic_shape + [4])
        process_clses = tf.zeros(basic_shape + [20])

        [i, raw_clses, raw_bboxes, score, process_bboxes, process_clses] = \
            tf.while_loop(condition, body, [i, raw_clses, raw_bboxes, score, process_bboxes, process_clses])

        process_label = tf.concat([score, process_bboxes, process_clses], axis=-1)

        return process_label


def _process_image(raw_img, raw_bboxes, **kwargs):

    with tf.variable_scope('preprocess_image'):

        if raw_img.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        if not raw_img.dtype == tf.float32:
            raw_img = tf.to_float(raw_img)
            raw_img = raw_img / 255.

        process_img = tf.image.resize_images(raw_img, [kwargs['img_size'], kwargs['img_size']],
                                       method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=False)
        # augmentation
        process_img, raw_bboxes = aug.random_flip_left_right(process_img, raw_bboxes)
        # todo random crop
        process_img = aug.distort_color(process_img)
        process_img = process_img * 255.

        return process_img, raw_bboxes


def preprocess_for_train(raw_img, raw_clses, raw_bboxes, **kwargs):

    # process image
    process_img , raw_bboxes = _process_image(raw_img, raw_bboxes, **kwargs)

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

    # process clses and bboxes
    process_label = _preprocess_boxes(raw_clses, raw_bboxes, **kwargs)

    return process_img, process_label
