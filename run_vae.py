import os
import sys
import time
import argparse
from load_data import *
from generate_data import *
from vae_model import *

FLAGS = None


def train():
    # Load data as 'tf.data.Dataset'
    data_loader = DataLoader(FLAGS)
    if FLAGS.dataset == "mnist":
        l_train_data, u_train_data, valid_data, _ = data_loader.load_semi_mnist(
            FLAGS.input_file_path
        )
    else:
        raise ValueError("Unsupported dataset!")

    # Determine the construction ration of the semi-batch
    l_ratio = FLAGS.labelled_size / data_loader.train_size
    l_batch_size = max(1, int(l_ratio * FLAGS.batch_size))
    u_batch_size = max(1, int((1 - l_ratio) * FLAGS.batch_size))

    # Build input pipeline while taking account of edge cases
    if data_loader.pure_unsupervised:
        l_batch_size = 0
        l_train_data = l_train_data \
            .apply(tf.contrib.data.shuffle_and_repeat(10000))
    else:
        l_train_data = l_train_data \
            .apply(tf.contrib.data.shuffle_and_repeat(10000)) \
            .batch(l_batch_size) \
            .prefetch(buffer_size=l_batch_size)
    l_train_iterator = l_train_data.make_one_shot_iterator()

    if data_loader.pure_supervised:
        l_batch_size = FLAGS.batch_size
        u_next_element = None
    else:
        u_train_data = u_train_data \
            .apply(tf.contrib.data.shuffle_and_repeat(10000)) \
            .batch(u_batch_size) \
            .prefetch(buffer_size=u_batch_size)
        u_iterator = u_train_data.make_one_shot_iterator()
        u_next_element = u_iterator.get_next()

    valid_data = valid_data\
        .batch(FLAGS.batch_size) \
        .prefetch(buffer_size=FLAGS.batch_size)
    valid_iterator = valid_data.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    l_iterator = tf.data.Iterator.from_string_handle(
        handle,
        valid_data.output_types,
        valid_data.output_shapes
    )
    l_next_element = l_iterator.get_next()

    # Build computational graph
    model = VAE(FLAGS,
                data_loader.pure_supervised,
                data_loader.pure_unsupervised)
    model.build_model(l_next_element[0], l_next_element[1], u_next_element)

    # Prepare for training session
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Launch a session
    with tf.Session(config=config) as sess:
        # Initialize parameters or load them from ckpt file
        if FLAGS.restore:
            print("Reading checkpoints ...")
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_para_path)
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
            except:
                raise ValueError("Fail to load ckpt!")
        else:
            sess.run(tf.global_variables_initializer())

        # Initialize tf.data.Iterator
        train_handle = sess.run(l_train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())

        # Create summary writer
        train_writer = tf.summary.FileWriter(
            FLAGS.log_path + 'train',
            sess.graph
        )
        valid_writer = tf.summary.FileWriter(
            FLAGS.log_path + 'valid'
        )

        # Main Training Loop
        for train_step in range(FLAGS.num_epoch * data_loader.train_size
                                // FLAGS.batch_size):
            epoch_id = train_step * FLAGS.batch_size // data_loader.train_size

            start_time = time.time()
            _, train_loss = sess.run([model.train_op, model.L], feed_dict={
                handle: train_handle,
                model.l_batch_size: l_batch_size
            })
            elapsed_time = time.time() - start_time

            print("Step {} / Epoch {} : Loss {:.2f} ({:.4f} ms)"
                  .format(train_step, epoch_id, train_loss, elapsed_time))

            if train_step % FLAGS.log_freq == 0:
                summary = sess.run(model.merged_summary, feed_dict={
                    handle: train_handle,
                    model.l_batch_size: l_batch_size
                })
                train_writer.add_summary(summary, train_step)

            if train_step % FLAGS.eval_freq == 0:
                sess.run(valid_iterator.initializer)
                total_valid_loss = 0
                total_accuracy = 0
                summary = None

                for valid_step in range(data_loader.valid_size
                                        // FLAGS.batch_size):
                    if summary is None:
                        valid_loss, accuracy, summary = sess.run(
                            [model.L, model.accuracy, model.merged_summary],
                            feed_dict={handle: valid_handle,
                                       model.l_batch_size: FLAGS.batch_size}
                        )
                    else:
                        valid_loss, accuracy = sess.run(
                            [model.L, model.accuracy],
                            feed_dict={handle: valid_handle,
                                       model.l_batch_size: FLAGS.batch_size}
                        )
                    total_valid_loss += valid_loss * FLAGS.batch_size
                    total_accuracy += accuracy * FLAGS.batch_size

                valid_writer.add_summary(summary, train_step)
                print("[Validation] Loss {:.2f} Accuracy {:.4f}"
                      .format(total_valid_loss / data_loader.valid_size,
                              total_accuracy / data_loader.valid_size))

            if train_step % FLAGS.save_freq == 0:
                saver.save(sess,
                           os.path.join(FLAGS.model_para_path, 'model'),
                           global_step=train_step)

            '''
            if train_step % FLAGS.visual_freq == 0:
                save_as_images_grid(
                    "{}epoch{}_truth".format(
                        FLAGS.result_path,
                        train_step * FLAGS.batch_size // data_loader.train_size
                    ),
                    x_truth[:FLAGS.batch_size],
                    FLAGS.batch_size,
                    10
                )
                save_as_images_grid(
                    "{}epoch{}_decode".format(
                        FLAGS.result_path,
                        train_step * FLAGS.batch_size // data_loader.train_size
                    ),
                    x_gen[:FLAGS.batch_size],
                    FLAGS.batch_size,
                    10
                )
            '''


def generate():
    # Build computational graph
    model = VAE(FLAGS, pure_unsupervised=True)
    model.build_model(
        tf.constant(0., shape=[]),
        tf.constant(0., shape=[]),
        tf.constant(0., shape=[FLAGS.batch_size, FLAGS.dim_x])
    )

    # Sample z from standard gaussian distribution
    z = np.random.normal(loc=0.0, scale=1.0,
                         size=(FLAGS.batch_size, FLAGS.dim_z))

    # Launch a session
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Restore parameters from ckpt
        print("Reading checkpoints ...")
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_para_path)
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
        except:
            raise ValueError("Fail to load ckpt!")

        # Generate x_hat from randomly sampled z
        x_gen = sess.run(model.x_gen_sigmoid, feed_dict={model.z: z,
                                                         model.l_batch_size: 0})

        # Save as pic
        save_as_images_grid(
            "{}generate_label{}".format(FLAGS.result_path, FLAGS.labelled_size),
            x_gen,
            FLAGS.batch_size,
            10
        )


def test_or_encode():
    # Compute the prediction accuracy on test set
    def _test():
        total_accuracy = 0
        for test_step in range(data_loader.test_size // FLAGS.batch_size):
            accuracy = sess.run(model.accuracy, feed_dict={
                model.l_batch_size: FLAGS.batch_size
            })
            total_accuracy += accuracy * FLAGS.batch_size
        print("Accuracy on test set: {:.4f}."
              .format(total_accuracy / data_loader.test_size))

    # Encode x(from test set) into z and visualize z space
    def _encode():
        z_list, y_list = [], []
        for test_step in range(data_loader.test_size // FLAGS.batch_size):
            z_batch, y_batch = sess.run([model.z, tf.argmax(next_element[1])],
                                        feed_dict={
                                            model.l_batch_size: FLAGS.batch_size
                                        })
            z_list.append(z_batch)
            y_list.append(y_batch)

        z = np.vstack(z_list)
        y = np.vstack(y_list)
        visualize_z_space(FLAGS.result_path + "z_space", z, y)

    # Prepare for input pipeline using 'tf.data'
    data_loader = DataLoader(FLAGS)
    if FLAGS.dataset == "mnist":
        _, _, _, test_data = data_loader.load_semi_mnist(
            FLAGS.input_file_path
        )
    else:
        raise ValueError("Unsupported dataset!")

    test_data = test_data.batch(FLAGS.batch_size) \
        .prefetch(buffer_size=FLAGS.batch_size)
    test_iterator = test_data.make_one_shot_iterator()
    next_element = test_iterator.get_next()

    # Build computational graph
    model = VAE(FLAGS, pure_supervised=True, mute_class_loss=True)
    model.build_model(next_element[0], next_element[1], None)

    # Launch a session
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Restore parameters from ckpt
        print("Reading checkpoints ...")
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_para_path)
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
        except:
            raise ValueError("Fail to load ckpt!")

        if FLAGS.action == "test":
            _test()
        else:
            _encode()


def main(_):
    if FLAGS.action == "train":
        train()
    elif FLAGS.action == "generate":
        generate()
    elif FLAGS.action in ["test", "encode"]:
        test_or_encode()
    else:
        raise ValueError("Unsupported mode!")


def print_hyper_paras():
    for key, value in FLAGS.__dict__.items():
        print('{}={}'.format(key, value))


def make_dirs():
    for key, value in FLAGS.__dict__.items():
        if key.endswith("path"):
            dir_name = value.rpartition("/")[0]
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action',
        default='train',
        choices=['train', 'test', 'generate', 'encode'],
    )
    parser.add_argument(
        '--dataset',
        default='mnist',
        choices=['mnist'],
        help='define which dataset to use'
    )
    parser.add_argument(
        '--input_file_path',
        default='dataset/mnist_l{}_u{}.pkl.gz',
        help='the path of input data file'
    )
    parser.add_argument(
        '--dim_x',
        default=784,
        type=int
    )
    parser.add_argument(
        '--dim_y',
        default=10,
        type=int
    )
    parser.add_argument(
        '--dim_hid',
        default=500,
        type=int
    )
    parser.add_argument(
        '--dim_z',
        default=10,
        type=int
    )
    parser.add_argument(
        '--activation',
        default='sigmoid',
        choices=['sigmoid', 'relu'],
        help='the activation function of MLP'
    )
    parser.add_argument(
        '--num_layers',
        default=1,
        type=int
    )
    parser.add_argument(
        '--num_class_layers',
        default=3,
        type=int
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float
    )
    parser.add_argument(
        '--loss_lambda',
        default=1,
        type=float
    )
    parser.add_argument(
        '--num_epoch',
        default=10,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        default=100,
        type=int
    )
    parser.add_argument(
        '--labelled_size',
        default=0,
        type=int
    )
    parser.add_argument(
        '--unlabelled_size',
        default=40000,
        type=int
    )
    parser.add_argument(
        '--log_freq',
        default=100,
        type=int
    )
    parser.add_argument(
        '--eval_freq',
        default=100,
        type=int
    )
    parser.add_argument(
        '--save_freq',
        default=500,
        type=int
    )
    parser.add_argument(
        '--visual_freq',
        default=5000,
        type=int
    )
    parser.add_argument(
        '--restore',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--model_para_path',
        default='model/l{}_u{}/',
    )
    parser.add_argument(
        '--log_path',
        default='log/l{}_u{}'
    )
    parser.add_argument(
        '--result_path',
        default='result/l{}_u{}/'
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.input_file_path = FLAGS.input_file_path.format(FLAGS.labelled_size,
                                                         FLAGS.unlabelled_size)
    FLAGS.model_para_path = FLAGS.model_para_path.format(FLAGS.labelled_size,
                                                         FLAGS.unlabelled_size)
    FLAGS.log_path = FLAGS.log_path.format(FLAGS.labelled_size,
                                           FLAGS.unlabelled_size)
    FLAGS.result_path = FLAGS.result_path.format(FLAGS.labelled_size,
                                                 FLAGS.unlabelled_size)
    print_hyper_paras()
    make_dirs()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
