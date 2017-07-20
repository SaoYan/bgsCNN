# from generate_bg import generate_bg
from prepare_data import prepare_data

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim

# generate_bg()
# prepare_data(321, 321)

def weight(shape, name):
	initial = tf.random_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def bias(shape, name):
	initial = tf.constant(0, shape=shape, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')

def deconv2d(x, W, output_shape, strides):
	return tf.nn.conv2d_transpose(x, W, output_shape = output_shape, strides = strides, padding = 'VALID')

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
    inputs = np.ones([img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], 7], dtype = np.float32)
    outputs_gt = np.ones([img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], 1], dtype = np.float32)
    for i in range(num):
        input_cast = image[i,:,:,0:6].astype(dtype = np.float32)
        input_norm = cv2.normalize(input_cast, 0., 1., cv2.NORM_MINMAX)
        gt = image[i,:,:,6]
        idx = ((gt != 0) & (gt != 255))
        gt[idx] = 0
        gt_norm = cv2.normalize(gt, 0, 1, cv2.NORM_MINMAX)
        gt_cast = gt_norm.astype(dtype = np.float32)
        inputs[i,:,:,:] = input_norm
        outputs_gt[i,:,:,:] = gt_cast
    return inputs, outputs_gt


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("batch_size", 100, "size of training batch")
    tf.app.flags.DEFINE_integer("test_size", 1000, "# of test samples used for testing")
    tf.app.flags.DEFINE_integer("max_iteration", 10000, "maximum # of training steps")
    tf.app.flags.DEFINE_integer("image_height", 321, "height of inputs")
    tf.app.flags.DEFINE_integer("image_width", 321, "width of inputs")
    tf.app.flags.DEFINE_integer("image_depth", 8, "depth of inputs")

    with tf.name_scope("input_data"):
        frame_and_bg = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_height, 6])
        fg_gt = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_height, 1])

    with tf.name_scope("pre_conv"):
        W_pre = weight([1, 1, 6, 3], "weights")
        pre_conv = conv2d(frame_and_bg, W_pre)
        tf.summary.histogram("W_pre_conv", W_pre)

    with tf.name_scope("resnet_v2_152"):
    	with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    	    net, end_points = resnet_v2.resnet_v2_152(
    	        pre_conv,
    	        num_classes = None,
    	        is_training = True,
    	        global_pool = False,
                output_stride = 16)

    with tf.name_scope("deconv_1"):
        W_deconv1 = weight([1, 1, 1024, 2048], "weights")
        deconv_1 = deconv2d(net, W_deconv1,
            output_shape = [FLAGS.batch_size, 81, 81, 1024], strides = [1, 4, 4, 1])
        tf.summary.histogram("W_deconv1", W_deconv1)

    with tf.name_scope("deconv_2"):
        W_deconv2 = weight([5, 5, 64, 1024], "weights")
        deconv_2 = deconv2d(deconv_1, W_deconv2,
            output_shape = [FLAGS.batch_size, 165, 165, 64], strides = [1, 2, 2, 1])
        tf.summary.histogram("W_deconv2", W_deconv2)

    with tf.name_scope("deconv_3"):
        W_deconv3 = weight([5, 5, 16, 64], "weights")
        deconv_3 = deconv2d(deconv_2, W_deconv3,
            output_shape = [FLAGS.batch_size, 333, 333, 16], strides = [1, 2, 2, 1])
        tf.summary.histogram("W_deconv3", W_deconv3)

    with tf.name_scope("conv_1"):
        W_conv1 = weight([13, 13, 16, 4], "weights")
        conv_1 = conv2d(deconv_3, W_conv1)
        tf.summary.histogram("W_conv1", W_conv1)

    with tf.name_scope("conv_2"):
        W_conv2 = weight([1, 1, 4, 1], "weights")
        conv_2 = tf.nn.sigmoid(conv2d(conv_1, W_conv2))
        tf.summary.histogram("W_conv2", W_conv2)

    with tf.name_scope("evaluation"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = fg_gt, logits = conv_2))
        tf.summary.scalar("loss", cross_entropy)

    with tf.name_scope('training_op'):
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(cross_entropy)

    train_file = "train.tfrecords"
    test_file  = "test.tfrecords"
    saver = tf.train.Saver()
    img_size = [FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth]
    train_batch = tf.train.shuffle_batch([read_tfrecord(train_file, img_size)],
                batch_size = FLAGS.batch_size,
                capacity = 50000,
                num_threads = 4,
                min_after_dequeue = 10000)
    test_batch = tf.train.shuffle_batch([read_tfrecord(test_file, img_size)],
                batch_size = FLAGS.test_size,
                capacity = 50000,
                num_threads = 4,
                min_after_dequeue = 10000)
    init = tf.initialize_all_variables()
    init_fn = slim.assign_from_checkpoint_fn("CNN_models/resnet_v2_152.ckpt", slim.get_model_variables('resnet_v2'))
    with tf.Session() as sess:
        init_fn(sess)
        sess.run(init)
        summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./logs/train", sess.graph)
        test_writer  = tf.summary.FileWriter("./logs/test", sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        inputs_test, outputs_gt_test = build_img_pair(sess.run(test_batch))
        for iter in range(FLAGS.max_iteration):
            inputs_train, outputs_gt_train = build_img_pair(sess.run(train_batch))
            if iter%10 == 0:
                summary_train = sess.run(summary, {frame_and_bg:inputs_train, fg_gt:outputs_gt_train})
                train_writer.add_summary(summary_train, iter)
                train_writer.flush()
                summary_test = sess.run(summary, {frame_and_bg:inputs_test, fg_gt:outputs_gt_test})
                test_writer.add_summary(summary_test, iter)
                test_writer.flush()
            if iter%100 == 0:
                train_loss  = cross_entropy.eval({frame_and_bg:inputs_train, fg_gt:outputs_gt_train})
                test_loss   = cross_entropy.eval({frame_and_bg:inputs_test, fg_gt:outputs_gt_test})
                print("iter step %d trainning batch loss %f"%(iter, train_loss))
                print("iter step %d test loss %f"%(iter, test_loss))
            if iter%100 == 0:
                saver.save(sess, "./logs/model.ckpt", global_step=iter)
            train_step.run({frame_and_bg:inputs_train, fg_gt:outputs_gt_train})

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, "./logs/model.ckpt")
        test_loss = MSE_loss.eval({frame_and_bg:inputs_test, fg_gt:outputs_gt_test})
        print("final test loss %f" % test_loss)
