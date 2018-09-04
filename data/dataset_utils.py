import tensorflow as tf
import os

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (238, 306),
    'bicycle': (243, 353),
    'bird': (330, 486),
    'boat': (181, 290),
    'bottle': (244, 505),
    'bus': (186, 229),
    'car': (713, 1250),
    'cat': (337, 376),
    'chair': (445, 798),
    'cow': (141, 259),
    'diningtable': (200, 215),
    'dog': (421, 510),
    'horse': (287, 362),
    'motorbike': (245, 339),
    'person': (2008, 4690),
    'pottedplant': (245, 514),
    'sheep': (96, 257),
    'sofa': (229, 248),
    'train': (261, 297),
    'tvmonitor': (256, 324),
    'total': (5011, 12608),
}

TEST_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (1, 1),
    'bicycle': (1, 1),
    'bird': (1, 1),
    'boat': (1, 1),
    'bottle': (1, 1),
    'bus': (1, 1),
    'car': (1, 1),
    'cat': (1, 1),
    'chair': (1, 1),
    'cow': (1, 1),
    'diningtable': (1, 1),
    'dog': (1, 1),
    'horse': (1, 1),
    'motorbike': (1, 1),
    'person': (1, 1),
    'pottedplant': (1, 1),
    'sheep': (1, 1),
    'sofa': (1, 1),
    'train': (1, 1),
    'tvmonitor': (1, 1),
    'total': (20, 20),
}

SPLITS_TO_SIZES = {
    'train': 5011,
    'test': 4952,
}

SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}

NUM_CLASSES = 20


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

LABELS_FILENAME = 'labels.txt'

def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

    Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

    Returns:
    `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

    Returns:
    A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read()
    lines = lines.split(b'\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(b':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names