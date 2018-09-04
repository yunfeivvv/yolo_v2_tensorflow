import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from xml.etree import ElementTree as ET

from data.dataset_utils import _int64_feature, _float_feature, _bytes_feature
from data.dataset_utils import VOC_LABELS


def _process_image(dir, name):
    # image raw
    filename ='{}JPEGImages/{}.jpg'.format(dir, name)
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    # read xml file
    filename = '{}ImageSets/Annotations/{}.xml'.format(dir, name)
    tree = ET.parse(filename)
    root = tree.getroot()
    # shape
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # annotations
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, shape, bboxes, labels, labels_text, difficult, truncated):

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(shape[0]),
            'image/width': _int64_feature(shape[1]),
            'image/channels': _int64_feature(shape[2]),
            'image/shape': _int64_feature(shape),
            'image/object/bbox/xmin': _float_feature(xmin),
            'image/object/bbox/xmax': _float_feature(xmax),
            'image/object/bbox/ymin': _float_feature(ymin),
            'image/object/bbox/ymax': _float_feature(ymax),
            'image/object/bbox/label': _int64_feature(labels),
            'image/object/bbox/label_text': _bytes_feature(labels_text),
            'image/object/bbox/difficult': _int64_feature(difficult),
            'image/object/bbox/truncated': _int64_feature(truncated),
            'image/format': _bytes_feature(image_format),
            'image/encoded': _bytes_feature(image_data)}))
    return example


def convert_to_tfrecords(root_dir, save_dir, name='trainval', num_shards=4):
    assert name in ['all', 'train', 'val', 'trainval', 'test']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open ('{}ImageSets/Main/{}.txt'.format(root_dir, name), 'r') as f:
        all_ids = f.readlines()
        all_ids = [i.rstrip('\n') for i in all_ids]
        np.random.shuffle(all_ids)

    spacing = np.linspace(0., len(all_ids), num_shards + 1).astype(int)

    print('--- start ---')

    for shard in range(num_shards):
        print("Shard:[{0}/{1}]".format(shard+1, num_shards))
        saved_tfrecord_name = '{}{}.tfrecords-{:0>3d}of-{:0>3d}'.format(save_dir, name, shard, num_shards)

        with tf.python_io.TFRecordWriter(saved_tfrecord_name) as writer:
            now_ids = all_ids[spacing[shard]: spacing[shard + 1]]

            for id_ in tqdm(now_ids):
                image_data, shape, bboxes, labels, labels_text, difficult, truncated = _process_image(root_dir, id_)
                example = _convert_to_example(image_data, shape, bboxes, labels, labels_text, difficult, truncated)
                writer.write(example.SerializeToString())

    print('--- end ---')



if __name__ == '__main__':
    path_to_voc2007 = '../../../data/VOC2007/VOCdevkit/VOC2007/'
    convert_to_tfrecords(path_to_voc2007, './tfrecords/VOC2007/')