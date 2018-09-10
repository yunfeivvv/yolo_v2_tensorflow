import tensorflow as tf
import numpy as np
import cv2


def pred_to_hat(pred, **kwargs):
    '''
    pred 转换为 hat，即与gt相同的形式

    params:
        -* pred: [batch_size, 13, 13, 125]
        -* kwargs: cfg

    return:
        -* hat_scores: [batch_size, 13, 13, 5, 1]，
            两种解释：
                1、net预测当前cell当前anchor有目标的概率。
               *2、根据net预测的bboxes+当前ceil当前anchor的prior组成的“目标框”，和gt之间的IOU。
        -* hat_boxes:  [batch_size, 13, 13, 5, 4]，
            grid为单位长度时，bound boxes 在feature map上的位置。（yxhw）形式
        -* hat_clses:  [batch_szie, 13, 13, 5, 20]，
            取softmax就是预测的目标，因为20类不包括背景，所以要通过一个threshold。
    '''
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
            p_x, p_y = tf.meshgrid(tf.range(gs),# height
                                   tf.range(gs))# width
            p_x = tf.cast(tf.reshape(p_x, [1, gs, gs, 1, 1]), tf.float32)
            p_y = tf.cast(tf.reshape(p_y, [1, gs, gs, 1, 1]), tf.float32)
            p_yx = tf.concat([p_y, p_x], axis=4)
            p_hw = tf.reshape(kwargs['anchors'], [1, 1, 1, na, 2])

        with tf.variable_scope("get_hat"):
            hat_yx = (tf.sigmoid(pred_bbox_yx) + p_yx) / gs_f
            hat_hw = tf.exp(pred_bbox_hw) * p_hw / gs_f
            hat_bbox = tf.concat([hat_yx, hat_hw], axis=-1)
            hat_conf = tf.sigmoid(pred_conf)
            hat_cls = tf.sigmoid(pred_cls)

        with tf.variable_scope('hat_to_dense_raw'):
            # to yxyx
            ymid, xmid, height, width = tf.unstack(hat_bbox, axis=-1)
            ymin = ymid - height / 2.
            xmin = xmid - width / 2.
            ymax = ymid + height / 2.
            xmax = xmid + width / 2.
            yxyx_bbox = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

            #selected = tf.where(tf.greater(hat_conf, neg_thred))    # 4-D
            #scores = tf.gather(hat_conf,selected) #* tf.to_float(tf.greater(hat_conf, neg_thred))
            #clses = tf.gather(hat_cls,selected) * scores

            decoded_hat = tf.concat([hat_conf, yxyx_bbox, hat_cls*hat_conf], axis=-1)

    return decoded_hat


# dense_raw to sparse raw
# only for 1 image
def nms_tf(dense_raw, thred_iou=0.5, thred_prob=0.2, max_out=10, num_classes=20):
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


def nms_np(bboxes, clses, thred_iou=0.5, thred_prob=0.5):
    assert bboxes.shape[0] == clses.shape[0]
    assert bboxes.shape[1] == 4
    assert clses.shape[1] == 20
    num_bboxes = bboxes.shape[0]
    num_classes = clses.shape[1]

    for c in range(num_classes):
        cls = clses[..., c]
        sorted_indices = np.argsort(-cls)  # 由大到小

        for i in range(num_bboxes):
            idx_i = sorted_indices[i]

            if cls[idx_i] == 0:
                continue
            else:
                for j in range(i + 1, num_bboxes):
                    idx_j = sorted_indices[j]

                    if cal_iou_np(bboxes[idx_i], bboxes[idx_j]) >= thred_iou:
                        clses[idx_j, c] = 0.0

    max_clses_name = np.argmax(clses, axis=-1)
    max_clses_prob = np.max(clses, axis=-1)

    nms_clses_idx = np.where(max_clses_prob > thred_prob)
    nms_clses_name = np.reshape([max_clses_name[nms_clses_idx]], [-1, 1])  # name
    nms_clses_prob = np.reshape(max_clses_prob[nms_clses_idx], [-1, 1])  # prob
    nms_bboxes = bboxes[nms_clses_idx]  # bbox

    return np.concatenate([nms_clses_name, nms_clses_prob, nms_bboxes], axis=-1)

def cal_iou_np(bbox1, bbox2):  # ymin, xmin, ymax, xmax
    assert bbox1.size == 4
    assert bbox2.size == 4
    if not bbox1.ndim == 1:
        bbox1 = np.squeeze(bbox1)
    if not bbox2.ndim == 1:
        bbox2 = np.squeeze(bbox2)

    lu = np.maximum(bbox1[:2], bbox2[:2])
    rd = np.minimum(bbox1[2:], bbox2[2:])

    inter = np.maximum(0.0, rd - lu)
    inter_a = inter[0] * inter[1]

    hw1 = bbox1[2:] - bbox1[:2]
    a1 = hw1[0] * hw1[1]

    hw2 = bbox2[2:] - bbox2[:2]
    a2 = hw2[0] * hw2[1]

    union_a = a1 + a2 - inter_a
    iou = np.clip(inter_a / union_a, 0.0, 1.0)
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
