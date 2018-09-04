import tensorflow as tf
from net import yolo_v2, resnet_v2, mobile_net_v2
slim = tf.contrib.slim

# TODO mobile_net_v2暂时不能用

nets_map = {
    'yolo_v2':yolo_v2.yolo_v2,
    'resnet_v2_152': resnet_v2.resnet_v2_152_yolo,
    'resnet_v2_50': resnet_v2.resnet_v2_50_yolo,
    # 'mobile_net_v2': mobile_net_v2.mobilenet,
    #
}

arg_scopes_map = {
    'yolo_v2':yolo_v2.yolo_v2_arg_scope,
    'resnet_v2_152': resnet_v2.resnet_v2_arg_scope,
    'resnet_v2_50': resnet_v2.resnet_v2_arg_scope,
    # 'mobile_net_v2': mobile_net_v2.training_scope,
}

norm_map = {
    'yolo_v2':yolo_v2.yolo_v2_normalize,
    'resnet_v2_152':resnet_v2.resnet_v2_normalize,
    'resnet_v2_50': resnet_v2.resnet_v2_normalize,
    # 'mobile_net_v2': mobile_net_v2.mobilenet_normalize,
}

def get_net(name):

    assert name in arg_scopes_map
    assert name in nets_map

    net = nets_map[name]
    arg_scope = arg_scopes_map[name]
    norm = norm_map[name]

    def func(inputs, is_training, **kwarg):
        with slim.arg_scope(arg_scope() if not name=='mobile_net_v2' else arg_scope(is_training=is_training)):
            return net(norm(inputs), is_training, name, **kwarg)

    return func