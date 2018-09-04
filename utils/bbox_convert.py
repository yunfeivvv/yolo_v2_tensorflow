import tensorflow as tf
import numpy as np
import cv2


'''
raw -->(scale tran)->(sparse->dense)--> gt \
                                            \
                                             \
prior --> (iou) --> iou --> (相乘) --> mask --> ====>  loss
            ^                 ^               /
            |                 |              /
        gt bboxes          gt score         / 
                                           /
pred --(sig等)--> pfn --(相加)--------> hat/
                          ^
                          |  
                        prior

pred --(sig等)--> pfn --(相加)--> hat ---> (筛选) --> (scale) -->raw
                          ^                 ^
                          |                 |
                        prior           pred score



表示方式：
- prediction，       pred    模型输出
- prediction func,   pfn     模型输出通过一个限制函数，
- ground_truth，     gt      模型的label
- ground_truth_hat， hat     模型输出转为gt形式
- target，           tgt     label转成pred形式
- prior，            p       先验信息，用于pred和gt两种形式之间的转换。

注意：
1：不同表示的量纲并不相同：gt中IMG是单位长度，pred中GRID是单位长度，建议统一转为GRID为单位。
2：sigmoid(pred_yx) = pfn_yx, exp(pred_hw) = pfn_hw, 作为中间变量方便理解。

命名：
- pred_x, pred_y, pred_w, pred_h
- pfn_x,  pfn_y,  pfn_w,  pfn_h
- gt_x,   gt_y,   gt_w,   gt_h
# - tgt_x,  tgt_y,  tgt_w,  tgt_h
- hat_x,  hat_y,  hat_w,  hat_h
- p_x,    p_y,    p_w,    p_h

(pred_*对应论文中的t_*, p_*对应论文中的c_*/p_*, hat_*对应文中的b_*)
*
    hat_x = sig(pred_x) + p_x
    hat_y = sig(pred_y) + p_y
    hat_w = p_w * exp(pred_w)
    hat_h = p_h * exp(pred_h)
*
    # tgt_x = d_sig(gt_x - p_x)
    # tgt_y = d_sig(gt_y - p_y)
    # tgt_w = log(gt_w / p_w)
    # tgt_h = log(gt_h / p_h)
    # d_sig(x) = -log(1/x - 1)
'''


def pred_to_hat(pred, **kwargs):
    '''
    pred 转换为 hat，即与gt相同的形式

    params:
        -* pred: [batch_size, 13, 13, 125]
        -* kwargs: cfg

    return:
        -* hat_scores: [batch_size, 13, 13, 5, 1]，
            两种解释：
                1、net预测当前ceil当前anchor有目标的概率。
               *2、根据net预测的bboxes+当前ceil当前anchor的prior组成的“目标框”，和gt之间的IOU。
        -* hat_boxes:  [batch_size, 13, 13, 5, 4]，
            grid为单位长度时，bound boxes 在feature map上的位置。（yxhw）形式
        -* hat_clses:  [batch_szie, 13, 13, 5, 20]，
            取softmax就是预测的目标，因为20类不包括背景，所以要通过一个threshold。
    '''

    with tf.variable_scope("pred_to_hat"):

        with tf.variable_scope('pred_split'):
            pred = tf.reshape(pred, [-1, kwargs['grid_size'], kwargs['grid_size'],
                                     kwargs['num_anchors'], 5 + kwargs['num_classes']])
            pred_scores = pred[..., 0]
            pred_boxes = pred[..., 1:5]
            pred_clses = pred[..., 5:]

        with tf.variable_scope('prediction_function'):
            pfn_boxes_yx = tf.sigmoid(pred_boxes[..., 0:2])
            pfn_boxes_hw = tf.exp(pred_boxes[..., 2:4])

        with tf.variable_scope('prior'):
            p_x, p_y = tf.meshgrid([i for i in range(kwargs['grid_size'])],
                                   [i for i in range(kwargs['grid_size'])])
            p_x = tf.cast(tf.reshape(p_x, [kwargs['grid_size'], kwargs['grid_size'], 1, 1]), tf.float32)
            p_y = tf.cast(tf.reshape(p_y, [kwargs['grid_size'], kwargs['grid_size'], 1, 1]), tf.float32)
            p_yx = tf.concat([p_y, p_x], axis=3)
            p_hw = tf.reshape(tf.constant(kwargs['anchors']), [1, 1, 1, kwargs['num_anchors'], 2])

        with tf.variable_scope("get_hat"):
            hat_boxes_yx = pfn_boxes_yx + p_yx
            hat_boxes_hw = pfn_boxes_hw * p_hw
            hat_boxes = tf.concat([hat_boxes_yx, hat_boxes_hw], axis=-1)
            hat_scores = tf.sigmoid(pred_scores)  # pfn_scores就是hat_scores，详细见论文
            hat_clses = pred_clses

            hat = tf.concat([hat_scores, hat_boxes, hat_clses], axis=-1)

    return hat


def hat_to_dense_raw(hat, **kwargs):
    '''
    return: dense_raw：下一步送入NMS就可以了。
    '''

    with tf.variable_scope('hat_to_dense_raw'):

        hat_scores = hat[..., :1]
        hat_boxes = hat[..., 1:5]
        hat_clses = hat[..., 5:]

        scores = hat_scores * tf.to_float(tf.greater(hat_scores, kwargs['threshold_score']))

        yxhw_bboxes = hat_boxes / kwargs['grid_size']
        ymid, xmid, height, width = tf.unstack(yxhw_bboxes, axis=-1)
        ymin = ymid - height / 2.
        xmin = xmid - width / 2.
        ymax = ymid + height / 2.
        xmax = xmid + width / 2.
        yxyx_bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

        clses = tf.nn.softmax(hat_clses)
        clses = clses * scores  # clses里面包含很多0
        clses = clses * tf.to_float(tf.greater(clses, kwargs['threshold_score']))

        dense_raws = tf.concat([scores, yxyx_bboxes, clses], axis=-1)

    return dense_raws

# dense_raw to sparse raw
def tf_nms(dense_raw,  **kwargs):
    '''
    return:
        * selected_bboxes, (num_selected_bboxes, 4)
        * selected_clses,  (num_selected_bboxes,)
        * selected_probs,  (num_selected_bboxes,)
    '''

    # only for 1 image
    cond = tf.logical_or(tf.equal(dense_raw.shape[0], 1),
                         tf.equal(dense_raw.shape.ndims, 4))
    tf.Assert(cond, tf.shape(dense_raw))

    dense_scores = dense_raw[0, ..., :1]
    dense_boxes = dense_raw[0, ..., 1:5]
    dense_clses = dense_raw[0, ..., 5:]

    # flatten
    scores = tf.reshape(dense_scores, [-1])
    yxyx_bboxes = tf.reshape(dense_boxes, [-1, 4])
    clses = tf.reshape(dense_clses, [-1, kwargs['num_classes']])

    # NMS
    selected = []    # 每个元素对应一个类别
    for c in range(kwargs['num_classes']):
        cls = tf.squeeze(tf.gather(clses, c, axis=-1), axis=-1)
        selected_indices = tf.image.non_max_suppression(yxyx_bboxes, cls, tf.shape(cls)[0],
                                                        kwargs['threshold_iou2'],kwargs['threshold_score'], name='nms')

        selected_bbox = tf.gather(yxyx_bboxes, selected_indices, axis=0)
        selected_cls = tf.argmax(tf.gather(clses, selected_indices, axis=0), axis=1)
        selected_prob = tf.reduce_max(tf.gather(clses, selected_indices, axis=0), axis=1)

        selected.append(tf.concat([tf.expand_dims(selected_cls, axis=-1),
                                   tf.expand_dims(selected_prob, axis=-1),
                                   selected_bbox], axis=-1))    # N_selected_bboxes, 6
    selected = tf.concat(selected, axis=-1)

    return selected




VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
# selected_arr = sess.run(selected)







def np_nms(scores, boxes, clses, **kwargs):

    '''
    :param scores: [13, 13, 5]
    :param boxes:  [13, 13, 5, 4]
    :param clses:  [13, 13, 5, 20]
    :param kwargs:
    :return:
    '''
    scores = scores.flatt

    # suppress non-maximal boxes
    for c in range(kwargs['num_classes']):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes
