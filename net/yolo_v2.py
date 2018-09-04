import tensorflow as tf

slim = tf.contrib.slim


def yolo_v2_arg_scope():
    with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[3,3],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            weights_regularizer=slim.l2_regularizer(1.0),
                            ) as arg_sc:
            return arg_sc


def yolo_v2(inputs, is_training, scope='yolov2', **kwargs):

    num_outputs = (kwargs['num_classes'] + 5) * kwargs['num_anchors']
    with tf.variable_scope(scope, 'yolov2', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):

            net = slim.conv2d(inputs, 32, scope='c1')
            net = slim.max_pool2d(net, scope='p1')
            net = slim.conv2d(net, 64, scope='c2')
            net = slim.max_pool2d(net, scope='p2')
            net = slim.conv2d(net, 128, scope='c3')
            net = slim.conv2d(net, 64, kernel_size=[1, 1], scope='c4')
            net = slim.conv2d(net, 128, scope='c5')
            net = slim.max_pool2d(net, scope='p3')
            net = slim.conv2d(net, 256, scope='c6')
            net = slim.conv2d(net, 128, kernel_size=[1, 1], scope='c7')
            net = slim.conv2d(net, 256, scope='c8')
            net = slim.max_pool2d(net, scope='p4')
            net = slim.conv2d(net, 512, scope='c9')
            net = slim.conv2d(net, 256, kernel_size=[1, 1], scope='c10')
            net = slim.conv2d(net, 512, scope='c11')
            net = slim.conv2d(net, 256, kernel_size=[1, 1], scope='c12')
            net = slim.conv2d(net, 512, scope='c13')

            path_1 = tf.space_to_depth(net, block_size=2, name='path_1')

            net = slim.max_pool2d(net, scope='p5')
            net = slim.conv2d(net, 1024, scope='c14')
            net = slim.conv2d(net, 512, kernel_size=[1, 1], scope='c15')
            net = slim.conv2d(net, 1024, scope='c16')
            net = slim.conv2d(net, 512, kernel_size=[1, 1], scope='c17')
            net = slim.conv2d(net, 1024, scope='c18')

            # 前面的是分类网络的一部分
            net = slim.conv2d(net, 1024, scope='c19')
            net = slim.conv2d(net, 1024, scope='c20')
            path_2 = net
            net = tf.concat([path_1, path_2], 3, name='concat')
            net = slim.conv2d(net, 1024, scope='c21')
            net = slim.conv2d(net, num_outputs, kernel_size=[1, 1], scope='c22')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def yolo_v2_normalize(inputs):
    if not inputs.dtype == tf.float32:
        inputs = tf.to_float(inputs)
    inputs = inputs / 255.
    return inputs