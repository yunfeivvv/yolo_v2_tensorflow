import os
import csv
import yaml
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import cv2
slim = tf.contrib.slim
tf.set_random_seed(123)

from data.dataset_utils import VOC_LABELS
from net.net_factory import get_net
from utils.utils import finetune_init, restore_init, draw_bboxes
from utils.test_utils import decode_pred, nms_tf, nms_np


flags = tf.app.flags
flags.DEFINE_string('yml_path', './config.yml', '')
flags.DEFINE_string('img_path', './test_imgs/dog.jpg', '')
flags.DEFINE_float('thred_iou', 0.3, '')
flags.DEFINE_float('thred_prob', 0.3, '')

flags.DEFINE_string('net_name', 'resnet_v2_152',
                    'yolo_v2 / resnet_v2_50 / resnet_v2_152')

FLAGS = flags.FLAGS


def main(_):

    # load configs
    with open(FLAGS.yml_path, 'r', encoding='utf-8') as f:
        CONFIG = yaml.load(f.read())
    NET_cfg   = CONFIG['NET']
    PATH_cfg  = CONFIG['PATH']

    with tf.Graph().as_default():

        img_op = tf.placeholder(tf.float32, [1, None, None, 3])
        pred_op, end_points_op = get_net(FLAGS.net_name)(img_op, False, **NET_cfg)
        hat_op = decode_pred(pred_op, FLAGS.thred_prob, **NET_cfg)
        # 1 使用tf自带的nms
        # selected_op = nms_tf(hat_op,FLAGS.thred_iou, FLAGS.thred_prob, 10, NET_cfg['num_classes'])
        # 2 nms_np

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            restore_init(sess, PATH_cfg['restore'][FLAGS.net_name])

            img_arr = cv2.cvtColor(cv2.imread(FLAGS.img_path), cv2.COLOR_BGR2RGB)
            img_arr = cv2.resize(img_arr, (NET_cfg['img_size'], NET_cfg['img_size']))
            hat_arr = sess.run(hat_op, feed_dict={img_op:np.expand_dims(img_arr, 0)})   # ?x6 [prob,class,bbox]
            nms_arr = nms_np(hat_arr, FLAGS.thred_iou)
            img_with_pred = draw_bboxes(img_arr, nms_arr, VOC_LABELS, offset=1)

            plt.imshow(img_with_pred)
            plt.show()

if __name__ == "__main__":
    tf.app.run()
