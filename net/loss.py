import tensorflow as tf


def cal_iou(boxes1, boxes2):
    '''
    :param boxes1: (batch size, 13, 13, 5, 4), yxhw
    :param boxes2: (batch size, 13, 13, 5, 4), yhxw
    :return:
    '''
    # transform (y_mid, x_mid, h, w) to (y1, x1, y2, x2)
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)
    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)

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


'''
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

def yolo_loss(pred, gt, global_step, debug=True, scope='yolov2loss', **kwargs):

    with tf.variable_scope(scope, 'loss'):

        with tf.variable_scope('pred_split'):
            pred = tf.reshape(pred, [-1, kwargs['grid_size'], kwargs['grid_size'], kwargs['num_anchors'], 5+kwargs['num_classes']])
            pred_scores = pred[..., 0]
            pred_boxes = pred[..., 1:5]
            pred_clses = pred[..., 5:]

        with tf.variable_scope('prediction_function'):
            pfn_boxes_yx = tf.sigmoid(pred_boxes[..., 0:2])
            pfn_boxes_hw = tf.exp(pred_boxes[..., 2:4])

        with tf.variable_scope('label_split'):
            gt_scores = gt[..., 0]
            gt_boxes = gt[..., 1:5] * kwargs['grid_size'] # 统一量纲
            gt_clses = gt[..., 5:]

        with tf.variable_scope('prior'):
            p_x, p_y = tf.meshgrid([i for i in range(kwargs['grid_size'])],
                                   [i for i in range(kwargs['grid_size'])])
            p_x = tf.cast(tf.reshape(p_x, [kwargs['grid_size'], kwargs['grid_size'], 1, 1]), tf.float32)
            p_y = tf.cast(tf.reshape(p_y, [kwargs['grid_size'], kwargs['grid_size'], 1, 1]), tf.float32)
            p_yx = tf.concat([p_y, p_x], axis=3)
            p_hw = tf.reshape(tf.constant(kwargs['anchors']), [1, 1, 1, kwargs['num_anchors'], 2])

        with tf.variable_scope('pred_to_hat'):   # gt_hat_*
            hat_boxes_yx = pfn_boxes_yx + p_yx
            hat_boxes_hw = pfn_boxes_hw * p_hw
            hat_boxes = tf.concat([hat_boxes_yx, hat_boxes_hw], axis=-1)
            hat_scores = tf.sigmoid(pred_scores)    # pfn_scores就是hat_scores，详细见论文
            hat_clses = pred_clses

        with tf.variable_scope("compute_iou_hat_gt"):
            iou = cal_iou(hat_boxes, gt_boxes)      # (batch_size, GRID_size, GRID_size, NUM_ANCHORS)

        with tf.variable_scope('gt_to_pred'):   # tgt_*
            tgt_scores = iou * gt_scores
            # tgt_clses  = gt_clses
            tgt_clses = tf.argmax(gt_clses, axis=-1)    # sparse_softmax_cross_entropy_with_logits

        with tf.variable_scope('masks'):

            mask_cls = gt_scores * kwargs['lambda_cls']

            mask_score = tf.zeros_like(gt_scores)
            mask_score = mask_score + gt_scores * kwargs['lambda_obj']
            mask_score = mask_score + tf.to_float(iou < kwargs['threshold_iou1'])\
                                      * (1 - gt_scores) * kwargs['lambda_noobj']

            mask_coord = tf.expand_dims(gt_scores, axis=-1) * kwargs['lambda_coord']


        # Warm-up training
        # 训练初始，不要将_mask_no_boxes对应的tgt都置零。
        # seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        # seen = tf.assign_add(seen, 1.)

        mask_no_coord = tf.to_float(mask_coord < kwargs['lambda_coord'] / 2.)
        # seen = tf.assign_add(seen, 1.)

        gt_boxes_yx, gt_boxes_hw, mask_coord = tf.cond(tf.less(global_step, kwargs['warmup_steps'] + 1),
                                                       lambda: [gt_boxes[...,1:3]  + (0.5 + p_yx) * mask_no_coord,
                                                                gt_boxes[..., 3:5] + tf.ones_like(gt_boxes[...,3:5]) * \
                                                                tf.reshape(kwargs['anchors'], [1, 1, 1, kwargs['num_anchors'], 2]) * \
                                                                mask_no_coord,
                                                                tf.ones_like(mask_coord)],
                                                       lambda: [gt_boxes[...,1:3] ,
                                                                gt_boxes[..., 3:5],
                                                                mask_coord])

        with tf.variable_scope("class_loss"):
            nb_class_box = tf.reduce_sum(tf.to_float(mask_cls > 0.0))
            # loss_cls = tf.nn.softmax_cross_entropy_with_logits(labels=tgt_clses, logits=hat_clses)
            loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt_clses, logits=hat_clses)
            loss_cls = tf.reduce_sum(loss_cls * mask_cls) / (nb_class_box + 1e-6)

        with tf.variable_scope("score_loss"):
            nb_score_box = tf.reduce_sum(tf.to_float(mask_score > 0.0))
            loss_score = tf.reduce_sum(tf.square(tgt_scores - hat_scores) * mask_score) / (nb_score_box + 1e-6) / 2.

        with tf.variable_scope("coord_loss"):
            nb_coord_box = tf.reduce_sum(tf.to_float(mask_coord > 0.0))
            # loss_coord = tf.reduce_sum(tf.square(gt_boxes - hat_boxes) * mask_coord) / (nb_coord_box + 1e-6) / 2.
            loss_yx = tf.reduce_sum(tf.square(gt_boxes[...,1:3] - hat_boxes_yx) * mask_coord) / (nb_coord_box + 1e-6) / 2.
            loss_hw = tf.reduce_sum(tf.square(gt_boxes[...,3:5] - hat_boxes_hw) * mask_coord) / (nb_coord_box + 1e-6) / 2.

        with tf.variable_scope("REGULARIZATION_LOSSES"):
            loss_l2 = kwargs['weight_decay'] * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        loss = tf.cond(tf.less(global_step, kwargs['warmup_steps']),
                       lambda: tf.add_n([loss_cls, loss_score, loss_yx, loss_hw, loss_l2, 10.]),
                       lambda: tf.add_n([loss_cls, loss_score, loss_yx, loss_hw, loss_l2]))
        # loss = tf.add_n([loss_cls, loss_score, loss_yx, loss_hw, loss_l2])

        summary_loss = []
        if debug:
            nb_true_box = tf.reduce_sum(gt_scores)
            nb_pred_box = tf.reduce_sum(tf.to_float(tgt_scores > 0.5) * tf.to_float(hat_scores > 0.3))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_hw],    message='loss_hw    \t', summarize=1000)
            loss = tf.Print(loss, [loss_yx],    message='loss_yx    \t', summarize=1000)
            loss = tf.Print(loss, [loss_score], message='loss_score \t', summarize=1000)
            loss = tf.Print(loss, [loss_cls]  , message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss_l2]   , message='Loss L2    \t', summarize=1000)
            loss = tf.Print(loss, [loss]      , message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / tf.to_float(global_step)], message='Average Recall \t', summarize=1000)
            
            summary_loss = [tf.summary.scalar('loss_yx', loss_yx),
                            tf.summary.scalar('loss_hw', loss_hw),
                            tf.summary.scalar('loss_score', loss_score),
                            tf.summary.scalar('loss_cls', loss_cls),
                            tf.summary.scalar('loss_l2', loss_l2),
                            tf.summary.scalar('loss', loss),
                            tf.summary.scalar('current_recall', current_recall)]

    return loss, summary_loss