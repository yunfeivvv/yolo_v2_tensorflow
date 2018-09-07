import tensorflow as tf
from net import loss
slim = tf.contrib.slim

# 现在所有的模型都用相同的loss
losses_map = {
    'yolo_v2':loss.yolo_loss_v2,
    'yolo_v3':loss.yolo_loss_v3,
}

def get_loss(name, pred, gt, global_step, debug=True, **kwargs):

    if name not in losses_map:
        raise ValueError('Name: %s unknown' % name)

    return losses_map[name](pred, gt, global_step, debug, **kwargs)
