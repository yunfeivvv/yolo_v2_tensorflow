import tensorflow as tf
from data import dataset_utils

slim = tf.contrib.slim

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

SPLITS_TO_SIZES = {
    'all': 9963,
    'train': 2501,
    'val':2510,
    'trainval': 5011,
    'test': 4952,
}

NUM_CLASSES = 20


def get_split(split_name, dataset_dir, reader=None):
    return _get_split(split_name, dataset_dir, reader,
                      SPLITS_TO_SIZES, ITEMS_TO_DESCRIPTIONS, NUM_CLASSES)


def _get_split(split_name, dataset_dir, reader,
              split_to_sizes, items_to_descriptions, num_classes):
    # split_name 只能是 'train', 'val', 'test', 'trainval' 之一
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)

    tfrecords_sources = tf.gfile.Glob(dataset_dir+'%s.tfrecords*'%split_name)

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)


    return slim.dataset.Dataset(
            data_sources=tfrecords_sources,
            reader= tf.TFRecordReader if not reader else reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes)