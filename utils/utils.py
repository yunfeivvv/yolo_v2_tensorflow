import os
import cv2
import tensorflow as tf
slim = tf.contrib.slim

def save_cfg(output_dir, cfg):
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        cfg_dict = cfg.__dict__
        for key in sorted(cfg_dict.keys()):
            if key[0].isupper():
                cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                f.write(cfg_str)


def draw_bboxes(img, bboxes, labels, offset=1):
    img = img.copy()
    img_h, img_w = img.shape[:2]
    readable_map = {it[1][0]- offset: it[0] for it in labels.items()}

    for i in range(bboxes.shape[0]):
        cls = bboxes[i, 0]
        prob = bboxes[i, 1]
        bbox = bboxes[i, 2:]

        xmin = int(bbox[1] * img_w)
        ymin = int(bbox[0] * img_h)
        xmax = int(bbox[3] * img_w)
        ymax = int(bbox[2] * img_h)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        cv2.putText(img,
                    readable_map[cls] + ' ' + str(prob)[:6],
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * img_h,
                    (0, 0, 255), 1)
    return img


def finetune_init(sess, **kwargs):

    if kwargs['path'] is None:
        raise ValueError("Checkpoint that do not exist")

    exclusions = []
    if kwargs['checkpoint_exclude_scopes']:
        exclusions = [scope.strip() for scope in kwargs['checkpoint_exclude_scopes']]

    # variables_to_restore = []
    # for var in slim.get_model_variables():
    #     excluded = False
    #     for exclusion in exclusions:
    #         if var.op.name.startswith(exclusion):
    #             excluded = True
    #             break
    #     if not excluded:
    #         variables_to_restore.append(var)
    variables_to_restore = slim.get_variables_to_restore(exclude=exclusions)

    if kwargs['checkpoint_model_scope'] is not None:
        variables_to_restore = {
            var.op.name.replace(kwargs['model_name'], kwargs['checkpoint_model_scope']):
                var for var in variables_to_restore}

    # return slim.assign_from_checkpoint(kwargs['finetune_path'],
    #                                    variables_to_restore,
    #                                    ignore_missing_vars=False)
    loader = tf.train.Saver(variables_to_restore)
    loader.restore(sess, kwargs['path'])
    print("Fine tune from ", kwargs['path'])


def restore_init(sess, path):
    # restore
    load_vars = tf.global_variables()
    loader = tf.train.Saver(load_vars)
    if tf.gfile.IsDirectory(path):
        checkpoint_model = tf.train.get_checkpoint_state(path)
        checkpoint_load = checkpoint_model.model_checkpoint_path
        loader.restore(sess, checkpoint_load)
        print("Restore from :", checkpoint_load)
    else:
        loader.restore(sess, path)
        print("Restore from :", path)
