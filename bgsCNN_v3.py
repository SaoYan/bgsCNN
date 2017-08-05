# from generate_bg import generate_bg
# from prepare_data import prepare_data

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim

# generate_bg()
# total_num_train, total_num_test = prepare_data(321, 321)

def weight(shape, name):
	initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')

def deconv2d(x, W, output_shape, strides):
	return tf.nn.conv2d_transpose(x, W, output_shape = output_shape, strides = strides, padding = 'VALID')

def pool3d(x, ksize, strides, mode):
    x_pool = tf.transpose(x, perm=[0,3,1,2])
    x_pool = tf.expand_dims(x_pool, 4)
    if mode == 'avg':
        x_pool = tf.nn.avg_pool3d(x_pool, ksize, strides, 'VALID')
    if mode == 'max':
        x_pool = tf.nn.max_pool3d(x_pool, ksize, strides, 'VALID')
    x_pool = tf.squeeze(x_pool, [4])
    x_pool = tf.transpose(x_pool, perm=[0,2,3,1])
    return x_pool

def read_tfrecord(tf_filename, image_size):
    filename_queue = tf.train.string_input_producer([tf_filename])
    reader = tf.TFRecordReader()
    __, serialized_example = reader.read(filename_queue)

    feature={ 'image_raw': tf.FixedLenFeature([], tf.string) }
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, image_size)
    return image

def build_img_pair(img_batch):
    num = img_batch.shape[0]
    inputs = np.ones([img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], 6], dtype = np.float32)
    outputs_gt = np.ones([img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], 1], dtype = np.float32)
    for i in range(num):
        input_cast = img_batch[i,:,:,0:6].astype(dtype = np.float32)
        input_min = np.amin(input_cast)
        input_max = np.amax(input_cast)
        input_norm = (input_cast - input_min) / (input_max - input_min)

        gt = img_batch[i,:,:,6]
        idx = ((gt != 0) & (gt != 255))
        gt[idx] = 0
        gt_cast = gt.astype(dtype = np.float32)
        gt_min = np.amin(gt_cast)
        gt_max = np.amax(gt_cast)
        gt_norm = (gt_cast - gt_min) / (gt_max - gt_min)

        inputs[i,:,:,:] = input_norm
        outputs_gt[i,:,:,0] = gt_norm
    return inputs, outputs_gt

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("train_batch_size", 40, "size of training batch")
    tf.app.flags.DEFINE_integer("test_batch_size", 200, "size of test batch")
    tf.app.flags.DEFINE_integer("max_iteration", 10000, "maximum # of training steps")
    tf.app.flags.DEFINE_integer("image_height", 321, "height of inputs")
    tf.app.flags.DEFINE_integer("image_width", 321, "width of inputs")
    tf.app.flags.DEFINE_integer("image_depth", 7, "depth of inputs")

    with tf.name_scope("input_data"):
        frame_and_bg = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_height, 6])
        fg_gt = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_height, 1])
        learning_rate = tf.placeholder(tf.float32, [])
        batch_size = tf.placeholder(tf.int32, [])
        frame = tf.slice(frame_and_bg, [0,0,0,0], [-1,FLAGS.image_height, FLAGS.image_height, 3])
        bg = tf.slice(frame_and_bg, [0,0,0,3], [-1,FLAGS.image_height, FLAGS.image_height, 3])
        tf.summary.image("frame", frame, max_outputs=3)
        tf.summary.image("background", bg, max_outputs=3)
        tf.summary.image("groundtruth", fg_gt, max_outputs=3)

    with tf.name_scope("pre_conv"):
        # shape: 321X321X3
        W_pre = weight([1, 1, 6, 3], "weights")
        pre_conv = conv2d(frame_and_bg, W_pre)
        tf.summary.histogram("W_pre_conv", W_pre)
        tf.summary.image("pre_conv_out", pre_conv, max_outputs=3)

    with tf.name_scope("resnet_v2"):
        # shape: 21X21X2048
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            net, end_points = resnet_v2.resnet_v2_50(
                pre_conv,
                num_classes = None,
                is_training = True,
                global_pool = False,
                output_stride = 16)

    with tf.name_scope("feature_reduction"):
        # shape: 21X21X51
        net = pool3d(net, [1,48,1,1,1], [1,40,1,1,1], 'avg')

    with tf.name_scope("deconv_1"):
        # shape: 43X43X32
        W_deconv1 = weight([3, 3, 32, 51], "weights")
        deconv_1 = deconv2d(net, W_deconv1,
            output_shape = [batch_size, 43, 43, 32], strides = [1, 2, 2, 1])
        tf.summary.histogram("W_deconv1", W_deconv1)
        tf.summary.image("channel1", tf.slice(deconv_1, [0,0,0,0],[-1,43,43,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_1, [0,0,0,1],[-1,43,43,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_1, [0,0,0,2],[-1,43,43,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_1, [0,0,0,3],[-1,43,43,1]), max_outputs=3)

    with tf.name_scope("deconv_1_max_pooling"):
        # shape: 41X41X16
        deconv_1_pool = pool3d(deconv_1, [1,2,3,3,1], [1,2,1,1,1], 'max')
        tf.summary.image("channel1", tf.slice(deconv_1_pool, [0,0,0,0],[-1,41,41,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_1_pool, [0,0,0,1],[-1,41,41,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_1_pool, [0,0,0,2],[-1,41,41,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_1_pool, [0,0,0,3],[-1,41,41,1]), max_outputs=3)

    with tf.name_scope("deconv_2"):
        # shape: 83X83X8
        W_deconv2 = weight([3, 3, 8, 16], "weights")
        deconv_2 = deconv2d(deconv_1_pool, W_deconv2,
            output_shape = [batch_size, 83, 83, 8], strides = [1, 2, 2, 1])
        tf.summary.histogram("W_deconv2", W_deconv2)
        tf.summary.image("channel1", tf.slice(deconv_2, [0,0,0,0],[-1,83,83,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_2, [0,0,0,1],[-1,83,83,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_2, [0,0,0,2],[-1,83,83,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_2, [0,0,0,3],[-1,83,83,1]), max_outputs=3)

    with tf.name_scope("deconv_2_max_pooling"):
        # shape: 81X81X8
        deconv_2_pool = tf.nn.max_pool(deconv_2, [1,3,3,1], [1,1,1,1], 'VALID')
        tf.summary.image("channel1", tf.slice(deconv_2_pool, [0,0,0,0],[-1,81,81,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_2_pool, [0,0,0,1],[-1,81,81,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_2_pool, [0,0,0,2],[-1,81,81,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_2_pool, [0,0,0,3],[-1,81,81,1]), max_outputs=3)

    with tf.name_scope("deconv_3"):
        # shape: 163X163X4
        W_deconv3 = weight([3, 3, 4, 8], "weights")
        deconv_3 = deconv2d(deconv_2_pool, W_deconv3,
            output_shape = [batch_size, 163, 163, 4], strides = [1, 2, 2, 1])
        tf.summary.histogram("W_deconv3", W_deconv3)
        tf.summary.image("channel1", tf.slice(deconv_3, [0,0,0,0],[-1,163,163,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_3, [0,0,0,1],[-1,163,163,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_3, [0,0,0,2],[-1,163,163,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_3, [0,0,0,3],[-1,163,163,1]), max_outputs=3)

    with tf.name_scope("deconv_3_max_pooling"):
        # shape: 161X161X4
        deconv_3_pool = tf.nn.max_pool(deconv_3, [1,3,3,1], [1,1,1,1], 'VALID')
        tf.summary.image("channel1", tf.slice(deconv_3_pool, [0,0,0,0],[-1,161,161,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_3_pool, [0,0,0,1],[-1,161,161,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_3_pool, [0,0,0,2],[-1,161,161,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_3_pool, [0,0,0,3],[-1,161,161,1]), max_outputs=3)

    with tf.name_scope("deconv_4"):
        # shape: 323X323X1
        W_deconv4 = weight([3, 3, 1, 4], "weights")
        deconv_4 = deconv2d(deconv_3_pool, W_deconv4,
            output_shape = [batch_size, 323, 323, 1], strides = [1, 2, 2, 1])
        tf.summary.histogram("W_deconv4", W_deconv4)
        tf.summary.image("out", tf.slice(deconv_4, [0,0,0,0],[-1,323,323,1]), max_outputs=3)

    with tf.name_scope("deconv_4_max_pooling"):
        # shape: 321X321X1
        deconv_4_pool = tf.nn.max_pool(deconv_4, [1,3,3,1], [1,1,1,1], 'VALID')
        tf.summary.image("out", tf.slice(deconv_4_pool, [0,0,0,0],[-1,321,321,1]), max_outputs=3)

    with tf.name_scope("conv"):
        W_conv = weight([1, 1, 1, 1], "weights")
        conv = conv2d(deconv_4_pool, W_conv)
        tf.summary.histogram("W_conv1", W_conv)
        tf.summary.image("out", conv, max_outputs=3)

    with tf.name_scope("final_result"):
        output = tf.nn.sigmoid(conv)
        result = 255 * tf.cast(output + 0.5, tf.uint8)
        tf.summary.image("sigmoid_out", output, max_outputs=3)
        tf.summary.image("segmentation", result, max_outputs=3)

    with tf.name_scope("evaluation"):
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = fg_gt, logits = conv))
        tf.summary.scalar("loss", cross_entropy)

    with tf.name_scope('training_op'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_step = optimizer.minimize(cross_entropy)

    train_file = "train.tfrecords"
    test_file  = "test.tfrecords"
    saver = tf.train.Saver()
    img_size = [FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth]
    train_batch = tf.train.shuffle_batch([read_tfrecord(train_file, img_size)],
                batch_size = FLAGS.train_batch_size,
                capacity = 3000,
                num_threads = 2,
                min_after_dequeue = 1000)
    test_batch = tf.train.shuffle_batch([read_tfrecord(test_file, img_size)],
                batch_size = FLAGS.test_batch_size,
                capacity = 500,
                num_threads = 2,
                min_after_dequeue = 300)
    init = tf.global_variables_initializer()
    init_fn = slim.assign_from_checkpoint_fn("CNN_models/resnet_v2_50.ckpt", slim.get_model_variables('resnet_v2'))
    start_time = time.time()
    with tf.Session() as sess:
        init_fn(sess)
        sess.run(init)
        summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        test_writer  = tf.summary.FileWriter("logs/test", sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        inputs_test, outputs_gt_test = build_img_pair(sess.run(test_batch))
        for iter in range(FLAGS.max_iteration):
            inputs_train, outputs_gt_train = build_img_pair(sess.run(train_batch))
            # train with dynamic learning rate
            if iter <= 500:
                train_step.run({frame_and_bg:inputs_train, fg_gt:outputs_gt_train, learning_rate:1e-3, batch_size:FLAGS.train_batch_size})
            elif iter <= FLAGS.max_iteration - 1000:
                train_step.run({frame_and_bg:inputs_train, fg_gt:outputs_gt_train, learning_rate:0.5e-3, batch_size:FLAGS.train_batch_size})
            else:
                train_step.run({frame_and_bg:inputs_train, fg_gt:outputs_gt_train, learning_rate:1e-4, batch_size:FLAGS.train_batch_size})
            # print training loss and test loss
            if iter%10 == 0:
                summary_train = sess.run(summary, {frame_and_bg:inputs_train, fg_gt:outputs_gt_train, batch_size:FLAGS.train_batch_size})
                train_writer.add_summary(summary_train, iter)
                train_writer.flush()
                summary_test = sess.run(summary, {frame_and_bg:inputs_test, fg_gt:outputs_gt_test, batch_size:FLAGS.test_batch_size})
                test_writer.add_summary(summary_test, iter)
                test_writer.flush()
            # record training loss and test loss
            if iter%10 == 0:
                train_loss  = cross_entropy.eval({frame_and_bg:inputs_train, fg_gt:outputs_gt_train, batch_size:FLAGS.train_batch_size})
                test_loss   = cross_entropy.eval({frame_and_bg:inputs_test, fg_gt:outputs_gt_test, batch_size:FLAGS.test_batch_size})
                print("iter step %d trainning batch loss %f"%(iter, train_loss))
                print("iter step %d test loss %f\n"%(iter, test_loss))
            # record model
            if iter%100 == 0:
                saver.save(sess, "logs/model.ckpt", global_step=iter)
        # final result
        saver.save(sess, "logs/model.ckpt")
        final_test = 0
        for i in range(5):
            inputs_test, outputs_gt_test = build_img_pair(sess.run(test_batch))
            final_test = final_test + cross_entropy.eval({frame_and_bg:inputs_test, fg_gt:outputs_gt_test, batch_size:FLAGS.test_batch_size})
        final_test = final_test / 5.
        print("final test loss %f" % final_test)
        coord.request_stop()
        coord.join(threads)

        running_time = time.time() - start_time
        hour = int(running_time / 3600)
        minute = int((running_time % 3600) / 60)
        second = (running_time % 3600) % 60
        print("running time: %d h %d min %d sec" % (hour, minute, second))
