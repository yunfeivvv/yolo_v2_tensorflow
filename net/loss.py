import tensorflow as tf


def cal_iou(boxes1, boxes2):
    '''
    :param boxes1: (batch size, 13, 13, 5, 4), yxhw
    :param boxes2: (batch size, 13, 13, 5, 4), yxhw
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


def yolo_loss_v2(pred, gt, global_step, debug=True, scope='yolo_loss_v2', **kwargs):
    grid_size_f = tf.to_float(kwargs['grid_size'])
    with tf.variable_scope(scope):
        with tf.variable_scope('pred_split'):
            pred = tf.reshape(pred, [-1, kwargs['grid_size'], kwargs['grid_size'], kwargs['num_anchors'],
                                     5 + kwargs['num_classes']])
            pred_confs = pred[..., 0]
            pred_boxes = pred[..., 1:5]
            pred_clses = pred[..., 5:]

        with tf.variable_scope('prediction_function'):
            pfn_boxes_yx = tf.sigmoid(pred_boxes[..., 0:2])
            pfn_boxes_hw = tf.exp(pred_boxes[..., 2:4])
            pfn_confs    = tf.sigmoid(pred_confs)

        with tf.variable_scope('label_split'):
            gt_confs = gt[..., 0]
            gt_boxes = gt[..., 1:5]
            gt_clses = gt[..., 5:]

        with tf.variable_scope('prior'):
            p_x, p_y = tf.meshgrid([i for i in range(kwargs['grid_size'])],
                                   [i for i in range(kwargs['grid_size'])])
            p_x = tf.cast(tf.reshape(p_x, [1, kwargs['grid_size'], kwargs['grid_size'], 1, 1]), tf.float32)
            p_y = tf.cast(tf.reshape(p_y, [1, kwargs['grid_size'], kwargs['grid_size'], 1, 1]), tf.float32)
            p_yx = tf.concat([p_y, p_x], axis=-1)
            p_hw = tf.reshape(tf.constant(kwargs['anchors']), [1, 1, 1, kwargs['num_anchors'], 2])

        with tf.variable_scope('pred_to_hat'):  # gt_hat_*
            hat_boxes_yx = (pfn_boxes_yx + p_yx) / grid_size_f
            hat_boxes_hw = (pfn_boxes_hw * p_hw) / grid_size_f
            hat_boxes = tf.concat([hat_boxes_yx, hat_boxes_hw], axis=-1)
            hat_confs = pfn_confs
            hat_clses = pred_clses

        with tf.variable_scope("compute_iou_hat_gt"):
            iou = cal_iou(hat_boxes, gt_boxes)  # (batch_size, GRID_size, GRID_size, NUM_ANCHORS)
            # best_iou_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keepdims=True)))

        with tf.variable_scope('gt_to_pred'):
            tgt_confs = iou * gt_confs
            tgt_clses = tf.argmax(gt_clses, axis=-1)

        with tf.variable_scope('masks'):
            mask_cls = gt_confs * kwargs['lambda_cls']
            mask_conf = tf.zeros_like(gt_confs)
            mask_conf = mask_conf + gt_confs * kwargs['lambda_obj']
            mask_conf = mask_conf + tf.to_float(iou < kwargs['threshold_neg']) \
                                    * (1 - gt_confs) * kwargs['lambda_noobj']
            mask_coord = tf.expand_dims(gt_confs, axis=-1) * kwargs['lambda_coord']

        # Warm-up training
        total_recall = tf.Variable(0.)
        mask_no_coord = tf.to_float(mask_coord < kwargs['lambda_coord'] / 2.)

        gt_boxes_yx, gt_boxes_hw, mask_coord = tf.cond(tf.less(global_step, kwargs['warmup_steps'] + 1),
                                                       lambda: [gt_boxes[..., 0:2] + (0.5 + p_yx) * mask_no_coord / grid_size_f,
                                                                gt_boxes[..., 2:4] + tf.ones_like(gt_boxes[..., 2:4]) * \
                                                                tf.reshape(kwargs['anchors'],
                                                                           [1, 1, 1, kwargs['num_anchors'], 2]) * \
                                                                mask_no_coord / grid_size_f,
                                                                tf.ones_like(mask_coord)],
                                                       lambda: [gt_boxes[..., 0:2],
                                                                gt_boxes[..., 2:4],
                                                                mask_coord])
        # or
        # gt_boxes_yx, gt_boxes_hw = gt_boxes[..., 0:2], gt_boxes[..., 2:4]

        with tf.variable_scope("class_loss"):
            nb_class_box = tf.reduce_sum(tf.to_float(mask_cls > 0.0))
            loss_cls     = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tgt_clses, logits=hat_clses) * mask_cls) / (nb_class_box + 1e-6)
            # loss_cls = tf.reduce_sum(tf.square(gt_clses - tf.sigmoid(pred_clses)) * tf.expand_dims(mask_cls,axis=-1)) / (nb_class_box + 1e-6)

        with tf.variable_scope("conf_loss"):
            nb_conf_box = tf.reduce_sum(tf.to_float(mask_conf > 0.0))
            loss_conf   = tf.reduce_sum(tf.square(tgt_confs - hat_confs) * mask_conf) / (nb_conf_box + 1e-6) / 2.

        with tf.variable_scope("coord_loss"):
            nb_coord_box = tf.reduce_sum(tf.to_float(mask_coord > 0.0))
            loss_yx      = tf.reduce_sum(tf.square((gt_boxes_yx - hat_boxes_yx)) * mask_coord) / (
                nb_coord_box + 1e-6) / 2.
            loss_hw      = tf.reduce_sum(tf.square((gt_boxes_hw - hat_boxes_hw) / grid_size_f) *
                                         mask_coord) / ( nb_coord_box + 1e-6) / 2.
            # loss_hw      = tf.reduce_sum(tf.square(tf.sqrt(gt_boxes_hw) - tf.sqrt(hat_boxes_hw)) *
            #                              mask_coord) / ( nb_coord_box + 1e-6) / 2.

        with tf.variable_scope("regularize_loss"):
            loss_l2 = kwargs['weight_decay'] * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        loss = tf.cond(tf.less(global_step, kwargs['warmup_steps']),
                       lambda: tf.add_n([loss_cls, loss_conf, loss_yx, loss_hw, loss_l2, 10.]),
                       lambda: tf.add_n([loss_cls, loss_conf, loss_yx, loss_hw, loss_l2]))
        # loss = tf.add_n([loss_cls, loss_conf, loss_yx, loss_hw, loss_l2])

        summary_loss = []
        if debug:
            nb_true_box = tf.reduce_sum(gt_confs)
            nb_pred_box = tf.reduce_sum(tf.to_float(tgt_confs > 0.5) * tf.to_float(hat_confs > 0.3))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_hw],   message='loss_hw    \t', summarize=1000)
            loss = tf.Print(loss, [loss_yx],   message='loss_yx    \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='loss_conf  \t', summarize=1000)
            loss = tf.Print(loss, [loss_cls],  message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss_l2],   message='Loss L2    \t', summarize=1000)
            loss = tf.Print(loss, [loss],      message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall, nb_true_box, nb_pred_box], message='nb_pred_box \t', summarize=1000)
            loss = tf.Print(loss, [tf.reduce_sum(tf.to_float(tgt_confs > 0.5)), tf.reduce_sum(tf.to_float(hat_confs > 0.3))], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / tf.to_float(global_step)], message='Average Recall \t',
                            summarize=1000)

            summary_loss = [tf.summary.scalar('loss_yx', loss_yx),
                            tf.summary.scalar('loss_hw', loss_hw),
                            tf.summary.scalar('loss_conf', loss_conf),
                            tf.summary.scalar('loss_cls', loss_cls),
                            tf.summary.scalar('loss_l2', loss_l2),
                            tf.summary.scalar('loss', loss),
                            tf.summary.scalar('current_recall', current_recall)]

    return loss, summary_loss


def yolo_loss_v3(pred, gt, global_step, debug=True, scope='yolo_loss_v3', **kwargs):
    bs = pred.get_shape().as_list()[0]
    na = kwargs['num_anchors']
    nc = kwargs['num_classes']
    gs = kwargs['grid_size']
    gs_f = tf.to_float(gs)

    with tf.variable_scope(scope):
        with tf.variable_scope('prepare'):
            pred = tf.reshape(pred, [-1, gs, gs, na, 5 + nc])
            pred_conf = pred[:,:,:,:,0:1]
            pred_bbox = pred[:,:,:,:,1:5]
            pred_cls  = pred[:,:,:,:,5:5+nc]

            gt_conf = gt[:,:,:,:,0:1]
            gt_bbox = gt[:,:,:,:,1:5]
            gt_bbox_yx = gt_bbox[:,:,:,:,0:2]
            gt_bbox_hw = gt_bbox[:,:,:,:,2:4]
            gt_cls  = gt[:,:,:,:,5:5+nc]

            p_x, p_y = tf.meshgrid([i for i in range(gs)],
                                   [i for i in range(gs)])
            p_x = tf.cast(tf.reshape(p_x, [1, gs, gs, 1, 1]), tf.float32)
            p_y = tf.cast(tf.reshape(p_y, [1, gs, gs, 1, 1]), tf.float32)
            p_yx = tf.tile(tf.concat([p_y, p_x], axis=4), [bs, 1, 1, na, 1])
            p_hw = tf.tile(tf.reshape(kwargs['anchors'], [1, 1, 1, na, 2]), [bs, gs, gs, 1, 1])

        # pred to gt(hat)
        with tf.variable_scope('pred_to_hat'):
            hat_yx = (tf.sigmoid(pred_bbox[:,:,:,:,0:2]) + p_yx ) / gs_f
            hat_hw = tf.exp(pred_bbox[:,:,:,:,2:4]) * p_hw / gs_f
            hat_bbox = tf.concat([hat_yx, hat_hw], axis=-1)
            hat_conf = tf.sigmoid(pred_conf)
            hat_cls = tf.sigmoid(pred_cls)

        # mask
        with tf.variable_scope('mask'):
            #iou_anchor_gt = tf.expand_dims(cal_iou(tf.concat([p_yx/gs_f, p_hw/gs_f], axis=-1), gt_bbox), axis=-1)
            iou_pred_gt = tf.expand_dims(cal_iou(hat_bbox, gt_bbox), axis=-1)
            mask_noobj = tf.to_float(iou_pred_gt < kwargs['threshold_neg']) * (1-gt_conf)
            mask_obj = gt_conf
            mask_coord = gt_conf

        #warming up training
        with tf.variable_scope('warming_up'):
            mask_no_coord = tf.to_float(gt_conf < kwargs['lambda_coord'] / 2.)
            gt_bbox_yx, gt_bbox_hw, mask_coord = tf.cond(
                tf.less(global_step, kwargs['warmup_steps'] + 1),
                lambda: [gt_bbox_yx + (0.5 + p_yx) * mask_no_coord / gs_f,
                         gt_bbox_hw + tf.ones_like(gt_bbox_hw) * p_hw * mask_no_coord / gs_f,
                         tf.ones_like(mask_obj)],
                lambda: [gt_bbox_yx,
                         gt_bbox_hw,
                         mask_obj])

        with tf.variable_scope('loss'):
            num_pos = tf.reduce_sum(mask_obj)
            num_coo = tf.reduce_sum(mask_coord)
            num_neg = tf.reduce_sum(mask_noobj)

            loss_yx    = tf.reduce_sum(tf.square(hat_yx - gt_bbox_yx) * mask_coord) / (num_coo+1e-6) / 2.
            loss_hw    = tf.reduce_sum(tf.square(tf.sqrt(hat_hw) - tf.sqrt(gt_bbox_hw) * mask_coord)) / (num_coo+1e-6) / 2.
            loss_obj   = tf.reduce_sum(tf.square(hat_conf - 1     ) * mask_obj)   / (num_pos+1e-6) / 2.
            loss_noobj = tf.reduce_sum(tf.square(hat_conf - 0     ) * mask_noobj) / (num_neg+1e-6) / 2.
            loss_cls   = tf.reduce_sum(tf.square(hat_cls  - gt_cls) * mask_obj)   / (num_pos+1e-6) / 2.

            loss = loss_yx    * kwargs['lambda_coord'] + \
                   loss_hw    * kwargs['lambda_coord'] + \
                   loss_obj   * kwargs['lambda_obj']   + \
                   loss_noobj * kwargs['lambda_noobj'] + \
                   loss_cls   * kwargs['lambda_cls']

        with tf.variable_scope('summary'):
            ave_recall = tf.Variable(0.)
            n_p  = tf.reduce_sum(gt_conf)
            n_tp = tf.reduce_sum(tf.to_float(iou_pred_gt > 0.5) * tf.to_float(hat_conf > 0.3) * gt_conf)
            #n_tp = tf.reduce_sum(tf.to_float(hat_conf > 0.3) * gt_conf)

            now_recall = n_tp / (n_p + 1e-6)
            ave_recall = tf.assign_add(ave_recall, now_recall) / tf.to_float(global_step+1)

            if debug:
                loss = tf.Print(loss, [loss_hw],    message='loss_hw    \t', summarize=1000)
                loss = tf.Print(loss, [loss_yx],    message='loss_yx    \t', summarize=1000)
                loss = tf.Print(loss, [loss_obj],   message='loss_obj   \t', summarize=1000)
                loss = tf.Print(loss, [loss_noobj], message='loss_noobj \t', summarize=1000)
                loss = tf.Print(loss, [loss_cls],   message='Loss Class \t', summarize=1000)
                loss = tf.Print(loss, [loss],       message='Total Loss \t', summarize=1000)
                loss = tf.Print(loss, [now_recall], message='Now Recall \t', summarize=1000)
                loss = tf.Print(loss, [ave_recall], message='Ave Recall \t', summarize=1000)

            summary_loss = [tf.summary.scalar('loss_yx', loss_yx),
                            tf.summary.scalar('loss_hw', loss_hw),
                            tf.summary.scalar('loss_conf', loss_obj + loss_noobj),
                            tf.summary.scalar('loss_cls', loss_cls),
                            tf.summary.scalar('loss', loss),
                            tf.summary.scalar('recall', now_recall),
                            tf.summary.histogram('hat_hw', hat_hw),
                            tf.summary.histogram('hat_yx', hat_yx),
                            tf.summary.histogram('gt_bbox_hw', gt_bbox_hw),
                            tf.summary.histogram('gt_bbox_yx', gt_bbox_yx)]

        return loss, summary_loss
