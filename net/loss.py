import tensorflow as tf


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


# deprecated
def _cal_best_ious(pred, gt):

    def condition(i, pred, gt, best_ious):
        r = tf.less(i, tf.shape(gt)[1])                 # kwargs['max_num_bboxes_per_img']
        return r

    def body(i, pred, gt, best_ious):
        gt_bbox = gt[:,i,:]                             # Bx4, ymin, xmin, ymax, xmax
        gt_bbox = tf.reshape(gt_bbox, [-1, 1, 1, 1, 4]) # Bx1x1x1x4
        iou = cal_iou(pred, gt_bbox, type1='yxhw', type2='yxyx') # -> 16x13x13x5
        best_ious = tf.maximum(iou, best_ious)          #
        return [i+1, pred, gt, best_ious]

    i = 0
    best_ious = tf.zeros(tf.shape(pred)[:-1])
    [i, pred, gt, best_ious] = tf.while_loop(condition, body, [i, pred, gt, best_ious])

    return best_ious

# *
def _cal_best_ious_v2(hat_boxes, true_boxes):
    '''
    params:
        hat_bbox  : 16x13x13x5x4, yxhw 
        true_boxes: 16x20?x4,     yxyx
    return:
        best_ious : 16x13x13x5
    '''
    shape = tf.shape(true_boxes)				
    hat_boxes = tf.expand_dims(hat_boxes, axis=4)                           		# Bx13x13x 5x 1x4
    true_boxes = tf.reshape(true_boxes, [shape[0], 1, 1, 1, shape[1], shape[2]])    # Bx 1x 1x 1x20x4
    iou_scores = cal_iou(hat_boxes, true_boxes, type1='yxhw', type2='yxyx') 		# Bx13x13x 5x20
    best_ious = tf.reduce_max(iou_scores, axis=-1)                          		# Bx13x13x 5
    return best_ious


def yolo_loss_v3(pred, gt, true_boxes, global_step, debug=True, scope='yolo_loss_v3', **kwargs): 
    na = kwargs['num_anchors']
    nc = kwargs['num_classes']
    gs = kwargs['grid_size']
    gs_f = tf.to_float(gs)

    with tf.variable_scope(scope):
        with tf.variable_scope('prepare'):
            pred = tf.reshape(pred, [-1, gs, gs, na, 5 + nc])
            pred_conf = pred[:,:,:,:,0:1]
            pred_bbox = pred[:,:,:,:,1:5]
            pred_bbox_yx = pred_bbox[:,:,:,:,0:2]
            pred_bbox_hw = pred_bbox[:,:,:,:,2:4]
            pred_cls  = pred[:,:,:,:,5:5+nc]

            gt_conf = gt[:,:,:,:,0:1]
            gt_bbox = gt[:,:,:,:,1:5]
            gt_bbox_yx = gt_bbox[:,:,:,:,0:2]
            gt_bbox_hw = gt_bbox[:,:,:,:,2:4]
            gt_cls  = gt[:,:,:,:,5:5+nc]

            p_x, p_y = tf.meshgrid([i for i in range(gs)],# height
                                   [i for i in range(gs)])# width
            p_x = tf.cast(tf.reshape(p_x, [1, gs, gs, 1, 1]), tf.float32)
            p_y = tf.cast(tf.reshape(p_y, [1, gs, gs, 1, 1]), tf.float32)
            p_yx = tf.concat([p_y, p_x], axis=4)
            p_hw = tf.reshape(kwargs['anchors'], [1, 1, 1, na, 2])

        # pred to gt(hat)
        with tf.variable_scope('pred_to_hat'):
            pred_bbox_hw = tf.Print(pred_bbox_hw, [tf.reduce_mean(pred_bbox_hw)], message='mean of pred_bbox_hw: \t')
            hat_yx = (tf.sigmoid(pred_bbox_yx) + p_yx) / gs_f
            hat_hw = tf.exp(pred_bbox_hw) * p_hw / gs_f
            hat_bbox = tf.concat([hat_yx, hat_hw], axis=-1)
            #   if square error
            hat_conf = tf.sigmoid(pred_conf)
            #hat_cls = tf.sigmoid(pred_cls)
            # * if binary cross entropy or softmax
            #hat_conf = pred_conf
            hat_cls = pred_cls

        # gt to tgt
        #with tf.variable_scope('gt_to_tgt'):
        #    tgt_xy = gt_bbox_yx - p_yx / gs_f # fixme
        #    tgt_hw = tf.log(gt_bbox_hw / p_hw * gs_f)
        #    tgt_hw = tf.where(tf.tile(tf.cast(gt_conf, tf.bool), [1,1,1,1,2]), tgt_hw, tf.zeros_like(tgt_hw))

        # mask
        with tf.variable_scope('mask'):
            best_iou_pred_all_gt = tf.expand_dims(_cal_best_ious_v2(hat_bbox, true_boxes), axis=-1)
            mask_noobj = tf.to_float(best_iou_pred_all_gt < kwargs['thred_neg']) * (1-gt_conf)
            mask_obj = gt_conf
            mask_coord = gt_conf
            ext_coord_scale = 2. - gt_bbox_hw[..., 0:1] * gt_bbox_hw[..., 1:2]

        # warming up training
        with tf.variable_scope('warming_up'):
            gt_bbox_yx, gt_bbox_hw, mask_coord = tf.cond(
                tf.less(global_step, kwargs['warmup_steps'] + 1),
                lambda: [gt_bbox_yx + (0.5 + p_yx) * (1-mask_obj) / gs_f,
                         gt_bbox_hw + tf.ones_like(gt_bbox_hw) * p_hw * (1-mask_obj) / gs_f,
                         tf.ones_like(mask_obj)],
                lambda: [gt_bbox_yx,
                         gt_bbox_hw,
                         gt_conf])

        with tf.variable_scope('loss'):
            bs_f = tf.to_float(tf.shape(pred)[0])   # mean的时候用batch size 或者 num of bboxes
            num_pos = tf.maximum(tf.reduce_sum(mask_obj)  , 1.)
            num_coo = tf.maximum(tf.reduce_sum(mask_coord), 1.)
            num_neg = tf.maximum(tf.reduce_sum(mask_noobj), 1.)
            num_pos = tf.Print(num_pos, [num_pos, num_neg, num_coo], message='num_pos, num_neg, num_coo  \t')
            
            # coord loss:
            #loss_yx = tf.reduce_sum(tf.square((pred_bbox_yx - tgt_xy) * mask_coord * ext_coord_scale)) / bs_f * kwargs['lambda_coord']
            #loss_hw = tf.reduce_sum(tf.square((pred_bbox_hw - tgt_hw) * mask_coord * ext_coord_scale)) / bs_f * kwargs['lambda_coord']
            loss_yx = tf.reduce_sum(tf.square((hat_yx - gt_bbox_yx) * mask_coord * ext_coord_scale)) / num_coo * kwargs['lambda_coord']
            loss_hw = tf.reduce_sum(tf.square((hat_hw - gt_bbox_hw) * mask_coord * ext_coord_scale)) / num_coo * kwargs['lambda_coord']
            
            # confidence loss
            # * if square error
            obj_delta   = tf.square((hat_conf - 1.    ) * mask_obj  )
            noobj_delta = tf.square((hat_conf - 0.    ) * mask_noobj)
            #   if binary cross entropy
            #obj_delta   = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf, logits=hat_conf) * mask_obj
            #noobj_delta = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf, logits=hat_conf) * mask_noobj
            loss_obj    = tf.reduce_sum(obj_delta)   / num_pos * kwargs['lambda_obj']
            loss_noobj  = tf.reduce_sum(noobj_delta) / num_neg * kwargs['lambda_noobj']

            #  classfication loss
            #   if square error
            #cls_delta = tf.square(hat_cls  - gt_cls) * mask_obj
            #   if binary cross entropy
            #cls_delta = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cls , logits=hat_cls) * tf.squeeze(mask_obj,axis=-1)
            # * if softmax cross entropy
            cls_delta = tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls , logits=hat_cls) * tf.squeeze(mask_obj,axis=-1)
            loss_cls  = tf.reduce_sum(cls_delta) / num_pos * kwargs['lambda_cls']


            loss = loss_yx + loss_hw + loss_obj + loss_noobj + loss_cls #+ \
            # tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * kwargs['weight_decay']

        with tf.variable_scope('summary'):

            eval_iou = gt_conf * tf.expand_dims(cal_iou(hat_bbox, gt_bbox), axis=-1)   # only for eval
            # * if square
            #hat_conf_sig = hat_conf
            #   if binary entropy
            hat_conf_sig = tf.sigmoid(hat_conf)
            n_p  = tf.reduce_sum(gt_conf)
            n_f  = tf.reduce_sum(1-gt_conf)
            n_tp = tf.reduce_sum(hat_conf_sig * gt_conf)
            n_tf = tf.reduce_sum(hat_conf_sig * (1 - gt_conf))
            detect_mask = tf.to_float((hat_conf_sig * gt_conf) >= 0.5)
            class_mask = tf.expand_dims(tf.to_float(tf.equal(tf.arg_max(hat_cls,-1), tf.arg_max(gt_cls,-1))), -1)
            recall25 = tf.reduce_sum(class_mask * detect_mask * tf.to_float(eval_iou > 0.25)) / (n_p + 1e-6)
            recall50 = tf.reduce_sum(class_mask * detect_mask * tf.to_float(eval_iou > 0.50)) / (n_p + 1e-6)
            recall75 = tf.reduce_sum(class_mask * detect_mask * tf.to_float(eval_iou > 0.75)) / (n_p + 1e-6)

            avg_iou   = tf.reduce_sum(eval_iou) / (n_p + 1e-3)
            avg_obj   = tf.reduce_sum(hat_conf_sig * gt_conf) / (n_p + 1e-3)      # 有问题的
            avg_noobj = tf.reduce_sum(hat_conf_sig * (1 - gt_conf)) / (n_f + 1e-3)
            avg_cat   = tf.reduce_sum(gt_conf * class_mask) / (n_p + 1e-3)

            if debug:
                loss = tf.Print(loss, [avg_obj],  message='avg_obj  \t', summarize=1000)
                loss = tf.Print(loss, [avg_noobj],message='avg_noobj\t', summarize=1000)
                loss = tf.Print(loss, [avg_iou],  message='avg_iou  \t', summarize=1000)
                loss = tf.Print(loss, [avg_cat],  message='avg_cat  \t', summarize=1000)
                loss = tf.Print(loss, [recall25, recall50, recall75], 
                                      message='recall25 ,recall50, recall75 \t', summarize=1000)
                loss = tf.Print(loss, [loss_yx,loss_hw,loss_obj,loss_noobj,loss_cls], 
                                      message='loss yx, hw, obj, onobj, class: \t', summarize=1000)

            summary_loss = [tf.summary.scalar('loss_yx', loss_yx),
                            tf.summary.scalar('loss_hw', loss_hw),
                            tf.summary.scalar('loss_conf', loss_obj + loss_noobj),
                            tf.summary.scalar('loss_cls', loss_cls),
                            tf.summary.scalar('loss', loss),
                            tf.summary.scalar('recall50', recall50),
                            tf.summary.scalar('recall75', recall75),
                            tf.summary.scalar('n_p', n_p),
                            tf.summary.scalar('avg_cat', avg_cat),
                            tf.summary.scalar('avg_obj', avg_obj),
                            tf.summary.scalar('avg_noobj', avg_noobj),
                            tf.summary.scalar('avg_iou', avg_iou),

                            tf.summary.histogram('hat_conf', hat_conf),
                            #tf.summary.histogram('hat_hw', hat_hw),
                            #tf.summary.histogram('hat_yx', hat_yx),
                            #tf.summary.histogram('gt_bbox_hw', gt_bbox_hw),
                            #tf.summary.histogram('gt_bbox_yx', gt_bbox_yx),
                            ]

        return loss, summary_loss



# def yolo_loss_v2(pred, gt, global_step, debug=True, scope='yolo_loss_v2', **kwargs):
#     grid_size_f = tf.to_float(kwargs['grid_size'])
#     with tf.variable_scope(scope):
#         with tf.variable_scope('prepare'):
#             pred = tf.reshape(pred, [-1, kwargs['grid_size'], kwargs['grid_size'], kwargs['num_anchors'],
#                                      5 + kwargs['num_classes']])
#             pred_confs = pred[..., 0]
#             pred_boxes = pred[..., 1:5]
#             pred_clses = pred[..., 5:]
#
#             gt_confs = gt[..., 0]
#             gt_boxes = gt[..., 1:5]
#             gt_clses = gt[..., 5:]
#
#             p_x, p_y = tf.meshgrid([i for i in range(kwargs['grid_size'])],
#                                    [i for i in range(kwargs['grid_size'])])
#             p_x = tf.cast(tf.reshape(p_x, [1, kwargs['grid_size'], kwargs['grid_size'], 1, 1]), tf.float32)
#             p_y = tf.cast(tf.reshape(p_y, [1, kwargs['grid_size'], kwargs['grid_size'], 1, 1]), tf.float32)
#             p_yx = tf.concat([p_y, p_x], axis=-1)
#             p_hw = tf.reshape(tf.constant(kwargs['anchors']), [1, 1, 1, kwargs['num_anchors'], 2])
#
#         with tf.variable_scope('pred_to_hat'):  # gt_hat_*
#             hat_boxes_yx = tf.sigmoid(pred_boxes[..., 0:2]) + p_yx
#             hat_boxes_hw = tf.exp(pred_boxes[..., 2:4]) * p_hw
#             hat_boxes = tf.concat([hat_boxes_yx, hat_boxes_hw], axis=-1)
#             hat_confs = tf.sigmoid(pred_confs)
#             hat_clses = pred_clses
#
#         with tf.variable_scope('gt_to_pred'):
#             tgt_boxes = gt_boxes * grid_size_f
#             iou = cal_iou(hat_boxes, tgt_boxes)  # (batch_size, GRID_size, GRID_size, NUM_ANCHORS)
#             tgt_confs = iou * gt_confs
#             tgt_clses = tf.argmax(gt_clses, axis=-1)
#
#         with tf.variable_scope('masks'):
#             mask_cls = gt_confs * kwargs['lambda_cls']
#             mask_conf = gt_confs * kwargs['lambda_obj'] + \
#                         tf.to_float(iou < kwargs['threshold_neg']) * (1 - gt_confs) * kwargs['lambda_noobj']
#             mask_coord = tf.expand_dims(gt_confs, axis=-1) * kwargs['lambda_coord']
#
#         # Warm-up training
#         total_recall = tf.Variable(0.)
#         mask_no_coord = tf.to_float(mask_coord < kwargs['lambda_coord'] / 2.)
#
#         gt_boxes_yx, gt_boxes_hw, mask_coord = tf.cond(
#             tf.less(global_step, kwargs['warmup_steps'] + 1),
#             lambda: [gt_boxes[..., 0:2] + (0.5 + p_yx) * mask_no_coord,
#                      gt_boxes[..., 2:4] + tf.ones_like(gt_boxes[..., 2:4]) * p_hw * mask_no_coord ,
#                      tf.ones_like(mask_coord)],
#             lambda: [gt_boxes[..., 0:2],
#                      gt_boxes[..., 2:4],
#                      mask_coord])
#
#         with tf.variable_scope("class_loss"):
#             nb_class_box = tf.maximum(5., tf.reduce_sum(tf.to_float(mask_cls > 0.0)))
#             loss_cls     = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
#                 labels=tgt_clses, logits=hat_clses) * mask_cls) / (nb_class_box + 1e-6)
#
#         with tf.variable_scope("conf_loss"):
#             nb_conf_box = tf.maximum(5., tf.reduce_sum(tf.to_float(mask_conf > 0.0)))
#             loss_conf   = tf.reduce_sum(tf.square(tgt_confs - hat_confs) * mask_conf) / (nb_conf_box + 1e-6) / 2.
#
#         with tf.variable_scope("coord_loss"):
#             nb_coord_box = tf.reduce_sum(tf.to_float(mask_coord > 0.0))
#             loss_yx      = tf.reduce_sum(tf.square((gt_boxes_yx - hat_boxes_yx)) * mask_coord) / (
#                 nb_coord_box + 1e-6) / 2.
#             loss_hw      = tf.reduce_sum(tf.square((gt_boxes_hw - hat_boxes_hw) / grid_size_f) *
#                                          mask_coord) / ( nb_coord_box + 1e-6) / 2.
#
#         with tf.variable_scope("regularize_loss"):
#             loss_l2 = kwargs['weight_decay'] * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#
#         # total loss
#         loss = tf.cond(tf.less(global_step, kwargs['warmup_steps']),
#                        lambda: tf.add_n([loss_cls, loss_conf, loss_yx, loss_hw, loss_l2, 10.]),
#                        lambda: tf.add_n([loss_cls, loss_conf, loss_yx, loss_hw, loss_l2]))
#         # loss = tf.add_n([loss_cls, loss_conf, loss_yx, loss_hw, loss_l2])
#
#         summary_loss = []
#         if debug:
#             nb_true_box = tf.reduce_sum(gt_confs)
#             nb_pred_box = tf.reduce_sum(tf.to_float(tgt_confs > 0.5) * tf.to_float(hat_confs > 0.3))
#
#             current_recall = nb_pred_box / (nb_true_box + 1e-6)
#             total_recall = tf.assign_add(total_recall, current_recall)
#
#             loss = tf.Print(loss, [loss_hw],   message='loss_hw    \t', summarize=1000)
#             loss = tf.Print(loss, [loss_yx],   message='loss_yx    \t', summarize=1000)
#             loss = tf.Print(loss, [loss_conf], message='loss_conf  \t', summarize=1000)
#             loss = tf.Print(loss, [loss_cls],  message='Loss Class \t', summarize=1000)
#             loss = tf.Print(loss, [loss_l2],   message='Loss L2    \t', summarize=1000)
#             loss = tf.Print(loss, [loss],      message='Total Loss \t', summarize=1000)
#             loss = tf.Print(loss, [current_recall, nb_true_box, nb_pred_box], message='nb_pred_box \t', summarize=1000)
#             loss = tf.Print(loss, [tf.reduce_sum(tf.to_float(tgt_confs > 0.5)), tf.reduce_sum(tf.to_float(hat_confs > 0.3))], message='Current Recall \t', summarize=1000)
#             loss = tf.Print(loss, [total_recall / tf.to_float(global_step)], message='Average Recall \t',
#                             summarize=1000)
#
#             summary_loss = [tf.summary.scalar('loss_yx', loss_yx),
#                             tf.summary.scalar('loss_hw', loss_hw),
#                             tf.summary.scalar('loss_conf', loss_conf),
#                             tf.summary.scalar('loss_cls', loss_cls),
#                             tf.summary.scalar('loss_l2', loss_l2),
#                             tf.summary.scalar('loss', loss),
#                             tf.summary.scalar('current_recall', current_recall)]
#
#     return loss, summary_loss
