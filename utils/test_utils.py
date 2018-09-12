import tensorflow as tf
import numpy as np
import cv2

def decode_pred(pred, thred_prob, **kwargs):

    na = kwargs['num_anchors']
    nc = kwargs['num_classes']
    gs = kwargs['grid_size']
    gs_f = tf.to_float(gs)

    with tf.variable_scope("pred_to_hat"):

        with tf.variable_scope('pred_split'):
            pred = tf.reshape(pred, [-1, gs, gs, na, 5 + nc])
            pred_conf = pred[:,:,:,:,0:1]
            pred_bbox = pred[:,:,:,:,1:5]
            pred_bbox_yx = pred_bbox[:,:,:,:,0:2]
            pred_bbox_hw = pred_bbox[:,:,:,:,2:4]
            pred_cls  = pred[:,:,:,:,5:5+nc]

        with tf.variable_scope('prior'):
            p_x, p_y = tf.meshgrid([i for i in range(gs)],# height
                                   [i for i in range(gs)])# width
            p_x = tf.cast(tf.reshape(p_x, [1, gs, gs, 1, 1]), tf.float32)
            p_y = tf.cast(tf.reshape(p_y, [1, gs, gs, 1, 1]), tf.float32)
            p_yx = tf.concat([p_y, p_x], axis=4)
            p_hw = tf.reshape(kwargs['anchors'], [1, 1, 1, na, 2])

        with tf.variable_scope("get_hat"):
            hat_yx = (tf.sigmoid(pred_bbox_yx) + p_yx) / gs_f
            hat_hw = tf.exp(pred_bbox_hw) * p_hw / gs_f
            ymid, xmid = tf.unstack(hat_yx, axis=-1)
            height, width = tf.unstack(hat_hw, axis=-1)
            ymin = ymid - height / 2.
            xmin = xmid - width  / 2.
            ymax = ymid + height / 2.
            xmax = xmid + width  / 2.

            hat_bbox = tf.stack([ymin, xmin, ymax, xmax], axis=-1)              # 1x13x13x5x4
            hat_conf = tf.sigmoid(pred_conf)                                    # 1x13x13x5x1
            hat_cls  = tf.nn.softmax(pred_cls) * hat_conf                       # 1x13x13x5x20
            hat_cls_prob = tf.reduce_max(hat_cls, axis=-1, keepdims=True)       # 1x13x13x5x1
            hat_cls_name = tf.argmax(hat_cls, axis=-1)
            hat_cls_name = tf.expand_dims(tf.to_float(hat_cls_name), axis=-1)   # 1x13x13x5x1

        with tf.variable_scope('thred_prob'):
            mask = tf.greater_equal(hat_cls_prob, thred_prob)
            mask = mask[..., 0]
            selected_cls_name = tf.boolean_mask(hat_cls_name, mask)             # ?x1
            selected_cls_prob = tf.boolean_mask(hat_cls_prob, mask)             # ?x1
            selected_bbox     = tf.boolean_mask(hat_bbox, mask)                 # ?x4

        decoded_hat = tf.concat([selected_cls_name, selected_cls_prob, selected_bbox], axis=-1)

    return decoded_hat


def nms_tf(dense_raw, thred_iou=0.3, thred_prob=0.3, max_out=10, num_classes=20):
    '''
    return:
        * selected_bboxes, (num_selected_bboxes, 4)
        * selected_clses,  (num_selected_bboxes,)
        * selected_probs,  (num_selected_bboxes,)
    '''

    dense_scores = dense_raw[0, ..., :1]
    dense_boxes = dense_raw[0, ..., 1:5]
    dense_clses = dense_raw[0, ..., 5:]

    # flatten
    scores = tf.reshape(dense_scores, [-1]) # score在上一步已经嵌入cls中了
    yxyx_bboxes = tf.reshape(dense_boxes, [-1, 4])
    clses = tf.reshape(dense_clses, [-1, num_classes])    # 先验概率

    # NMS
    selected = []    # 每个元素对应一个类别
    for c in range(num_classes):
        cls = tf.gather(clses, c, axis=-1)
        selected_indices = tf.image.non_max_suppression(yxyx_bboxes, cls, max_out, thred_iou, thred_prob, name='nms')

        selected_bbox = tf.gather(yxyx_bboxes, selected_indices, axis=0)
        selected_prob = tf.gather(cls, selected_indices, axis=0)

        selected.append(tf.concat([tf.expand_dims(selected_prob, axis=-1), selected_bbox], axis=-1))    # N_selected_bboxes, 5
    selected = tf.stack(selected, axis=0)

    return selected


def nms_np(pred, thred_iou=0.3, num_classes=20):

    nms = []
    clses  = pred[:, 0]
    probs  = pred[:, 1]
    bboxes = pred[:, 2:6]

    for c in range(num_classes):
        mask = np.equal(clses, c)
        num = int(np.sum(mask))
        if not num: continue    # 如果这个类别没有，循环下一个类别
        prob = probs[mask]      # ?
        bbox = bboxes[mask]     # ?x4
        sorted_indices = np.argsort(-prob)  # 由大到小

        for i in range(num):
            idx_i = sorted_indices[i]

            if prob[idx_i] == -1:   # 如果是-1，表示被suppress了
                continue
            else:
                for j in range(i + 1, num):
                    idx_j = sorted_indices[j]
                    if cal_iou_np(bbox[idx_i], bbox[idx_j]) >= thred_iou:
                        prob[idx_j] = -1

        nms_mask = np.greater(prob, 0.0)
        num_prob = np.expand_dims(prob[nms_mask], -1)
        num_name = np.ones_like(num_prob) * c
        nms_bbox = bbox[nms_mask]
        nms.append(np.concatenate([num_name, num_prob, nms_bbox], axis=-1))

    return np.concatenate(nms, axis=0)


def cal_iou_np(bbox1, bbox2):  # ymin, xmin, ymax, xmax

    lu = np.maximum(bbox1[:2], bbox2[:2])
    rd = np.minimum(bbox1[2:], bbox2[2:])

    inter = np.maximum(0.0, rd - lu)
    inter_a = inter[0] * inter[1]

    hw1 = np.maximum(0.0, bbox1[2:] - bbox1[:2])
    a1 = hw1[0] * hw1[1]

    hw2 = np.maximum(0.0, bbox2[2:] - bbox2[:2])
    a2 = hw2[0] * hw2[1]

    union_a = np.maximum(a1 + a2 - inter_a, 1e-10)
    iou = inter_a / union_a
    return iou

# def find_cell(x):
#     c = x%5
#     x = x//5
#     clo = x%13
#     x = x//13
#     row = x%13
#     x = x//13
#     assert x == 0
#     print("row:{}, clown:{}, channel:{}".format(row, clo, c))
#     return row, clo, c
