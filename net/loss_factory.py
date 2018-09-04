import tensorflow as tf
from net import loss
slim = tf.contrib.slim

# 现在所有的模型都用相同的loss
losses_map = {
    'yolov2':loss.yolo_loss,
    'yolov1':loss.yolo_loss,
    'resnet_v2_50': loss.yolo_loss,
    'resnet_v2_152':loss.yolo_loss,
}

def get_loss(name, pred, gt, global_step, debug=True, **kwargs):

    if name not in losses_map:
        raise ValueError('Name: %s unknown' % name)

    return losses_map[name](pred, gt, global_step, debug, **kwargs)
