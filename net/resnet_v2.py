import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v2


resnet_v2_arg_scope = resnet_v2.resnet_arg_scope


def resnet_v2_152_yolo(inputs, is_training, scope='resnet_v2_152', **kwargs):

    num_outputs = (kwargs['num_classes'] + 5) * kwargs['num_anchors']
    net, end_points = resnet_v2.resnet_v2_152(inputs, None, is_training, False)

    #path1 = tf.space_to_depth(end_points['resnet_v2_152/block2'], 2, 'path1')
    #path2 = end_points['resnet_v2_152/block3']
    #net = tf.concat([path1, net], axis=-1)
    net = slim.conv2d(net, 1024, kernel_size=[3, 3], scope='pred/conv1')
    net = slim.conv2d(net, num_outputs, kernel_size=[1, 1], activation_fn=None, normalizer_fn=None, weights_initializer=tf.initializers.random_normal(stddev=0.01), scope='pred/conv2')

    return net, end_points

def resnet_v2_50_yolo(inputs, is_training, scope='resnet_v2_50', **kwargs):

    num_outputs = (kwargs['num_classes'] + 5) * kwargs['num_anchors']
    net, end_points = resnet_v2.resnet_v2_50(inputs, None, is_training, False)
    #path1 = tf.space_to_depth(end_points['resnet_v2_50/block2'], 2, 'path1')
    #path2 = end_points['resnet_v2_50/block3']
    #net = tf.concat([path1, net], axis=-1)
    net = slim.conv2d(net, 1024, kernel_size=[3, 3], scope='pred/conv1')
    net = slim.conv2d(net, num_outputs, kernel_size=[1, 1], activation_fn=None, normalizer_fn=None, weights_initializer=tf.initializers.random_normal(stddev=0.01), scope='pred/conv2')

    return net, end_points


def resnet_v2_normalize(inputs):
    if not inputs.dtype == tf.float32:
        inputs = tf.to_float(inputs)
    inputs = inputs - tf.reshape([103.939, 116.779, 123.68], [1, 1, 1, 3])
    return inputs




# if __name__ == '__main__':
#     inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
#     with slim.arg_scope(resnet_v2.resnet_arg_scope()):
#         # net, end_points = resnet_v2.resnet_v2_50(inputs, None, is_training=False, global_pool=False)
#         net, end_points = resnet_v2_50_yolo(inputs, is_training=False, num_classes=20, num_anchors=5)
#
#     for k,v in end_points.items():
#         print(k+'\t', '-->', v.shape)
#     print(net.shape)
#
#     #sess =  tf.Session()
#     #writer = tf.summary.FileWriter('resnet1521',sess.graph)
#     #writer.close()
#     #sess.close()
