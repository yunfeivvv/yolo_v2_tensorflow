import os
import csv
import yaml
import tensorflow as tf
slim = tf.contrib.slim


from data.dataset_factory import get_data
from utils.processing import preprocess_for_train
from utils.mode_select import get_lr, get_optim
from net.net_factory import get_net
from net.loss_factory import get_loss
from utils.utils import finetune_init, restore_init


flags = tf.app.flags
flags.DEFINE_string('yml_path', './config.yml', '')
flags.DEFINE_string('start_with', 'finetune', 'finetune / restore')

flags.DEFINE_string('net_name', 'resnet_v2_50',
                    'yolo_v2 / resnet_v2_50 / resnet_v2_152')
flags.DEFINE_string('loss_name', 'yolo_v3',
                    'yolo_v2/yolo_v3/')
flags.DEFINE_string('lr_name', 'exp',
                    'constant/piecewise/exp/polynominal')
flags.DEFINE_string('optim_name', 'mom',
                    'sgd/mom/rms/adagrad/adadelta/adam')
flags.DEFINE_string('data_device', '/cpu:0', '')

FLAGS = flags.FLAGS


def main(_):
    symbol = None   # time.strftime('%m-%d-%H-%M-%S', time.localtime())

    # load configs
    with open(FLAGS.yml_path, 'r', encoding='utf-8') as f:
        CONFIG = yaml.load(f.read())

    TRAIN_cfg = CONFIG['TRAIN']
    NET_cfg   = CONFIG['NET']
    LR_cfg    = CONFIG['LR']
    OPTIM_cfg = CONFIG['OPTIM']
    PATH_cfg  = CONFIG['PATH']
    LOSS_cfg  = CONFIG['LOSS']
    GPU_cfg   = CONFIG['GPU']

    with tf.Graph().as_default():

        # =================================================================== #
        #                          load train data                            #
        # =================================================================== #
        with tf.device(FLAGS.data_device):
            dataset = get_data(PATH_cfg['train_data']['name'],
                               PATH_cfg['train_data']['split'],
                               PATH_cfg['train_data']['dir'])

            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=TRAIN_cfg['num_readers'],
                common_queue_capacity=20 * TRAIN_cfg['batch_size'],
                common_queue_min=10 * TRAIN_cfg['batch_size'],
                shuffle=True)

            [raw_img, raw_clses, raw_bboxes] = provider.get(['image', 'object/label', 'object/bbox'])

            # process
            train_img, train_gt = preprocess_for_train(raw_img, raw_clses, raw_bboxes, **NET_cfg)
            # batch
            train_imgs, train_gts = tf.train.batch([train_img, train_gt],
                                                   batch_size=TRAIN_cfg['batch_size'],
                                                   num_threads=TRAIN_cfg['num_readers'],
                                                   capacity=5*TRAIN_cfg['batch_size'])

        # =================================================================== #
        #                          compute graph                              #
        # =================================================================== #
        # inference
        train_preds, train_end_points = get_net(FLAGS.net_name)(train_imgs, True, **NET_cfg)

        # loss
        global_step = tf.Variable(0, trainable=False, name='global_step')
        loss, summary_loss = get_loss(FLAGS.loss_name, train_preds, train_gts, global_step, True, **{**NET_cfg, **LOSS_cfg})

        # train op
        learning_rate = get_lr(FLAGS.lr_name, global_step, **LR_cfg[FLAGS.lr_name])
        optimizer = get_optim(FLAGS.optim_name, learning_rate, **OPTIM_cfg[FLAGS.optim_name])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step, update_ops) # , clip_gradient_norm=5.0

        # summary
        train_summarys = []
        train_summarys.append(tf.summary.scalar('learning_rate', learning_rate))
        train_summarys.extend(summary_loss)
        train_merge = tf.summary.merge(train_summarys)

        # writer
        train_writer = tf.summary.FileWriter(PATH_cfg['log_dir'] + 'train/', tf.get_default_graph())


        # =================================================================== #
        #                            Session run                              #
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_cfg['gpu_memory_fraction'],
                                    allow_growth=GPU_cfg['allow_growth'])
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                allow_soft_placement=GPU_cfg['allow_soft_placement']))
        with sess.as_default():
            # save
            save_vars = tf.global_variables()
            saver = tf.train.Saver(save_vars)

            # init
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            if FLAGS.start_with == 'finetune':
                finetune_init(sess, **PATH_cfg['finetune'][FLAGS.net_name])
            elif FLAGS.start_with == 'restore':
                restore_init(sess, **PATH_cfg['restore'])

            # 开始训练
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)

            train_history = {'loss':[],
                             'step':[]}
            try:
                print('====== Start Training ======')

                for ep in range(1, TRAIN_cfg['epoches']+1):
                    for xx in range(TRAIN_cfg['ep_size']):
                        l, m, s, _ = sess.run([loss, train_merge, global_step, train_op])

                        train_history['loss'].append(l)
                        train_history['step'].append(s)
                        train_writer.add_summary(m, global_step=s)
                        # print("Epoch:{:4d}, loss:{:.4f}".format(ep, l))

                    # todo add vis fn

                    if (ep + 1) % TRAIN_cfg['save_ep'] == 0:
                        saver.save(sess, PATH_cfg['checkpoint_dir'] + FLAGS.net_name+'.ckpt', global_step=ep)


            except tf.errors.OutOfRangeError:
                print("catch OutOfRangeError")

            finally:

                print('====== End ======')
                coord.request_stop()
                print("finish reading")

                metric_dir = os.path.join(PATH_cfg['log_dir'], 'metric')
                if not os.path.exists(metric_dir):
                    os.makedirs(metric_dir)

                with open(os.path.join(metric_dir, 'loss.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'train_loss'])
                    for a, b in zip(train_history['step'], train_history['loss']):
                        writer.writerow([a, b])

            coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
