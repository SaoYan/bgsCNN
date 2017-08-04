# from generate_bg import generate_bg
from prepare_data import prepare_data

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim

# generate_bg()
total_num_train, total_num_test = prepare_data(320, 320)

def weight(shape, name):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial,name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding ='SAME')

def deconv2d(x, W, output_shape, padding):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,1,1,1], padding=padding)

def pooling(x):
    x_pool, indices = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    return x_pool, indices

def unpooling(x, indices,  shape):
    X = tf.reshape(x, [-1])
    Idx = rf.reshape(indices, [-1])
    N = np.prod(shape, dtype=np.int32)
    unpool = tf.Variable(tf.zeros([N]))
    unpool = tf.scatter_update(unpool, Idx, X)
    unpool = tf.reshape(unpool, shape=shape)
    return unpool

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
    tf.app.flags.DEFINE_integer("batch_size", 40, "size of training batch")
    tf.app.flags.DEFINE_integer("max_iteration", 10000, "maximum # of training steps")
    tf.app.flags.DEFINE_integer("image_height", 320, "height of inputs")
    tf.app.flags.DEFINE_integer("image_width", 320, "width of inputs")
    tf.app.flags.DEFINE_integer("image_depth", 7, "depth of inputs")

    with tf.name_scope("input_data"):
        frame_and_bg = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_height, 6])
        fg_gt = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_height, 1])
        learning_rate = tf.placeholder(tf.float32, [])
        frame = tf.slice(frame_and_bg, [0,0,0,0], [-1,FLAGS.image_height, FLAGS.image_height, 3])
        bg = tf.slice(frame_and_bg, [0,0,0,3], [-1,FLAGS.image_height, FLAGS.image_height, 3])
        tf.summary.image("frame", frame, max_outputs=3)
        tf.summary.image("background", bg, max_outputs=3)
        tf.summary.image("groundtruth", fg_gt, max_outputs=3)

    with tf.name_scope("conv_1"):
        # shape: 320X320X64
        W_conv1_1 = weight([3, 3, 6, 64], "weights_1")
        W_conv1_2 = weight([3, 3, 64, 64], "weights_2")
        conv_1_1 = tf.nn.relu(conv2d(frame_and_bg, W_conv1_1))
        conv_1_2 = tf.nn.relu(conv2d(conv_1_1, W_conv1_2))
        tf.summary.histogram("W_conv1_1", W_conv1_1)
        tf.summary.histogram("W_conv1_2", W_conv1_2)
        tf.summary.image("channel1", tf.slice(conv_1_2, [0,0,0,0],[-1,320,320,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_1_2, [0,0,0,1],[-1,320,320,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_1_2, [0,0,0,2],[-1,320,320,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_1_2, [0,0,0,3],[-1,320,320,1]), max_outputs=3)

    with tf.name_scope("conv_1_max_pooling"):
        # shape: 160X160X64
        conv_1_pool, indices_1 = pooling(conv_1_2)
        tf.summary.image("channel1", tf.slice(conv_1_pool, [0,0,0,0],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_1_pool, [0,0,0,1],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_1_pool, [0,0,0,2],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_1_pool, [0,0,0,3],[-1,160,160,1]), max_outputs=3)

    with tf.name_scope("conv_2"):
        # shape: 160X160x128
        W_conv2_1 = weight([3, 3, 64, 128], "weights_1")
        W_conv2_2 = weight([3, 3, 128, 128], "weights_2")
        conv_2_1 = tf.nn.relu(conv2d(conv_1_pool, W_conv2_1))
        conv_2_2 = tf.nn.relu(conv2d(conv_2_1, W_conv2_2))
        tf.summary.histogram("W_conv2_1", W_conv2_1)
        tf.summary.histogram("W_conv2_2", W_conv2_2)
        tf.summary.image("channel1", tf.slice(conv_2_2, [0,0,0,0],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_2_2, [0,0,0,1],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_2_2, [0,0,0,2],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_2_2, [0,0,0,3],[-1,160,160,1]), max_outputs=3)

    with tf.name_scope("conv_2_max_pooling"):
        # shape: 80X80X128
        conv_2_pool, indices_2 = pooling(conv_2_2)
        tf.summary.image("channel1", tf.slice(conv_2_pool, [0,0,0,0],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_2_pool, [0,0,0,1],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_2_pool, [0,0,0,2],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_2_pool, [0,0,0,3],[-1,80,80,1]), max_outputs=3)

    with tf.name_scope("conv_3"):
        # shape: 80X80X256
        W_conv3_1 = weight([3, 3, 128, 256], "weights_1")
        W_conv3_2 = weight([3, 3, 256, 256], "weights_2")
        W_conv3_3 = weight([3, 3, 256, 256], "weights_3")
        conv_3_1 = tf.nn.relu(conv2d(conv_2_pool, W_conv3_1))
        conv_3_2 = tf.nn.relu(conv2d(conv_3_1, W_conv3_2))
        conv_3_3 = tf.nn.relu(conv2d(conv_3_2, W_conv3_3))
        tf.summary.histogram("W_conv3_1", W_conv3_1)
        tf.summary.histogram("W_conv3_2", W_conv3_2)
        tf.summary.histogram("W_conv3_3", W_conv3_3)
        tf.summary.image("channel1", tf.slice(conv_3_3, [0,0,0,0],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_3_3, [0,0,0,1],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_3_3, [0,0,0,2],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_3_3, [0,0,0,3],[-1,80,80,1]), max_outputs=3)

    with tf.name_scope("conv_3_max_pooling"):
        # shape: 40X40X256
        conv_3_pool, indices_3 = pooling(conv_3_3)
        tf.summary.image("channel1", tf.slice(conv_3_pool, [0,0,0,0],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_3_pool, [0,0,0,1],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_3_pool, [0,0,0,2],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_3_pool, [0,0,0,3],[-1,40,40,1]), max_outputs=3)

    with tf.name_scope("conv_4"):
        # shape: 40X40X512
        W_conv4_1 = weight([3, 3, 256, 512], "weights_1")
        W_conv4_2 = weight([3, 3, 512, 512], "weights_2")
        W_conv4_3 = weight([3, 3, 512, 512], "weights_3")
        conv_4_1 = tf.nn.relu(conv2d(conv_3_pool, W_conv4_1))
        conv_4_2 = tf.nn.relu(conv2d(conv_4_1, W_conv4_2))
        conv_4_3 = tf.nn.relu(conv2d(conv_4_2, W_conv4_3))
        tf.summary.histogram("W_conv4_1", W_conv4_1)
        tf.summary.histogram("W_conv4_2", W_conv4_2)
        tf.summary.histogram("W_conv4_3", W_conv4_3)
        tf.summary.image("channel1", tf.slice(conv_4_3, [0,0,0,0],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_4_3, [0,0,0,1],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_4_3, [0,0,0,2],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_4_3, [0,0,0,3],[-1,40,40,1]), max_outputs=3)

    with tf.name_scope("conv_4_max_pooling"):
        # shape: 20X20X512
        conv_4_pool, indices_4 = pooling(conv_4_3)
        tf.summary.image("channel1", tf.slice(conv_4_pool, [0,0,0,0],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_4_pool, [0,0,0,1],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_4_pool, [0,0,0,2],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_4_pool, [0,0,0,3],[-1,40,40,1]), max_outputs=3)

    with tf.name_scope("conv_5"):
        # shape: 20X20X512
        W_conv5_1 = weight([3, 3, 512, 512], "weights_1")
        W_conv5_2 = weight([3, 3, 512, 512], "weights_2")
        W_conv5_3 = weight([3, 3, 512, 512], "weights_3")
        conv_5_1 = tf.nn.relu(conv2d(conv_4_pool, W_conv5_1))
        conv_5_2 = tf.nn.relu(conv2d(conv_5_1, W_conv5_2))
        conv_5_3 = tf.nn.relu(conv2d(conv_5_2, W_conv5_3))
        tf.summary.histogram("W_conv5_1", W_conv5_1)
        tf.summary.histogram("W_conv5_2", W_conv5_2)
        tf.summary.histogram("W_conv5_3", W_conv5_3)
        tf.summary.image("channel1", tf.slice(conv_5_3, [0,0,0,0],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_5_3, [0,0,0,1],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_5_3, [0,0,0,2],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_5_3, [0,0,0,3],[-1,20,20,1]), max_outputs=3)

    with tf.name_scope("conv_5_max_pooling"):
        # shape: 10X10X512
        conv_5_pool, indices_5 = pooling(conv_5_3)
        tf.summary.image("channel1", tf.slice(conv_5_pool, [0,0,0,0],[-1,10,10,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(conv_5_pool, [0,0,0,1],[-1,10,10,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(conv_5_pool, [0,0,0,2],[-1,10,10,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(conv_5_pool, [0,0,0,3],[-1,10,10,1]), max_outputs=3)

    with tf.name_scope("fully_connect"):
        # shape: 1x1x1024
        W_fc = weight([10,10,512,1024], "weight")
        fc = tf.nn.conv2d(conv_5_pool, W_fc, strides=[1,1,1,1], padding ='VALID')
        tf.summary.histogram("W_fc", W_fc)

    with tf.name_scope("de_fc"):
        # shape: 10X10X512
        W_defc = weight([10,10,512,1024], "weight")
        defc = deconv2d(fc, W_defc, output_shape=[FLAGS.batch_size,10,10,512], padding='VALID')
        tf.summary.histogram("W_defc", W_defc)
        tf.summary.image("channel1", tf.slice(defc, [0,0,0,0],[-1,10,10,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(defc, [0,0,0,1],[-1,10,10,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(defc, [0,0,0,2],[-1,10,10,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(defc, [0,0,0,3],[-1,10,10,1]), max_outputs=3)

    with tf.name_scope("unpool_5"):
        # shape: 20X20X512
        unpool_5 = unpooling(defc, indices_5, shape=[FLAGS.batch_size,20,20,512])
        tf.summary.image("channel1", tf.slice(unpool_5, [0,0,0,0],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(unpool_5, [0,0,0,1],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(unpool_5, [0,0,0,2],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(unpool_5, [0,0,0,3],[-1,20,20,1]), max_outputs=3)

    with tf.name_scope("deconv_5"):
        # shape: 20X20X512
        W_deconv5_1 = weight([3, 3, 512, 512], "weights_1")
        W_deconv5_2 = weight([3, 3, 512, 512], "weights_2")
        W_deconv5_3 = weight([3, 3, 512, 512], "weights_3")
        deconv_5_1 = tf.nn.relu(deconv2d(unpool_5, W_deconv5_1, output_shape=[FLAGS.batch_size,20,20,512], padding='SAME'))
        deconv_5_2 = tf.nn.relu(deconv2d(deconv_5_1, W_deconv5_2, output_shape=[FLAGS.batch_size,20,20,512], padding='SAME'))
        deconv_5_3 = tf.nn.relu(deconv2d(deconv_5_2, W_deconv5_3, output_shape=[FLAGS.batch_size,20,20,512], padding='SAME'))
        tf.summary.histogram("W_conv5_1", W_deconv5_1)
        tf.summary.histogram("W_conv5_2", W_deconv5_2)
        tf.summary.histogram("W_conv5_3", W_deconv5_3)
        tf.summary.image("channel1", tf.slice(deconv_5_3, [0,0,0,0],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_5_3, [0,0,0,1],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_5_3, [0,0,0,2],[-1,20,20,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_5_3, [0,0,0,3],[-1,20,20,1]), max_outputs=3)

    with tf.name_scope("unpool_4"):
        # shape: 40X40X512
        unpool_4 = unpooling(deconv_5_3, indices_4, shape=[FLAGS.batch_size,40,40,512])
        tf.summary.image("channel1", tf.slice(unpool_4, [0,0,0,0],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(unpool_4, [0,0,0,1],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(unpool_4, [0,0,0,2],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(unpool_4, [0,0,0,3],[-1,40,40,1]), max_outputs=3)

    with tf.name_scope("deconv_4"):
        # shape: 40X40X256
        W_deconv4_1 = weight([3, 3, 512, 512], "weights_1")
        W_deconv4_2 = weight([3, 3, 512, 512], "weights_2")
        W_deconv4_3 = weight([3, 3, 256, 512], "weights_3")
        deconv_4_1 = tf.nn.relu(deconv2d(unpool_4, W_deconv4_1, output_shape=[FLAGS.batch_size,40,40,512], padding='SAME'))
        deconv_4_2 = tf.nn.relu(deconv2d(deconv_4_1, W_deconv4_2, output_shape=[FLAGS.batch_size,40,40,512], padding='SAME'))
        deconv_4_3 = tf.nn.relu(deconv2d(deconv_4_2, W_deconv4_3, output_shape=[FLAGS.batch_size,40,40,256], padding='SAME'))
        tf.summary.histogram("W_conv4_1", W_deconv4_1)
        tf.summary.histogram("W_conv4_2", W_deconv4_2)
        tf.summary.histogram("W_conv4_3", W_deconv4_3)
        tf.summary.image("channel1", tf.slice(deconv_4_3, [0,0,0,0],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_4_3, [0,0,0,1],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_4_3, [0,0,0,2],[-1,40,40,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_4_3, [0,0,0,3],[-1,40,40,1]), max_outputs=3)

    with tf.name_scope("unpool_3"):
        # shape: 80X80X256
        unpool_3 = unpooling(deconv_4_3, indices_3, shape=[FLAGS.batch_size,80,80,256])
        tf.summary.image("channel1", tf.slice(unpool_3, [0,0,0,0],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(unpool_3, [0,0,0,1],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(unpool_3, [0,0,0,2],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(unpool_3, [0,0,0,3],[-1,80,80,1]), max_outputs=3)

    with tf.name_scope("deconv_3"):
        # shape: 80X80X128
        W_deconv3_1 = weight([3, 3, 256, 256], "weights_1")
        W_deconv3_2 = weight([3, 3, 256, 256], "weights_2")
        W_deconv3_3 = weight([3, 3, 128, 256], "weights_3")
        deconv_3_1 = tf.nn.relu(deconv2d(unpool_3, W_deconv3_1, output_shape=[FLAGS.batch_size,80,80,256], padding='SAME'))
        deconv_3_2 = tf.nn.relu(deconv2d(deconv_3_1, W_deconv3_2, output_shape=[FLAGS.batch_size,80,80,256], padding='SAME'))
        deconv_3_3 = tf.nn.relu(deconv2d(deconv_3_2, W_deconv3_3, output_shape=[FLAGS.batch_size,80,80,128], padding='SAME'))
        tf.summary.histogram("W_conv3_1", W_deconv3_1)
        tf.summary.histogram("W_conv3_2", W_deconv3_2)
        tf.summary.histogram("W_conv3_3", W_deconv3_3)
        tf.summary.image("channel1", tf.slice(deconv_3_3, [0,0,0,0],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_3_3, [0,0,0,1],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_3_3, [0,0,0,2],[-1,80,80,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_3_3, [0,0,0,3],[-1,80,80,1]), max_outputs=3)

    with tf.name_scope("unpool_2"):
        # shape: 160X160X128
        unpool_2 = unpooling(deconv_3_3, indices_2, shape=[FLAGS.batch_size,160,160,128])
        tf.summary.image("channel1", tf.slice(unpool_2, [0,0,0,0],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(unpool_2, [0,0,0,1],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(unpool_2, [0,0,0,2],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(unpool_2, [0,0,0,3],[-1,160,160,1]), max_outputs=3)

    with tf.name_scope("deconv_2"):
        # shape: 160X160x64
        W_deconv2_1 = weight([3, 3, 128, 128], "weights_1")
        W_deconv2_2 = weight([3, 3, 64, 128], "weights_2")
        deconv_2_1 = tf.nn.relu(deconv2d(unpool_2, W_deconv2_1, output_shape=[FLAGS.batch_size,160,160,128], padding='SAME'))
        deconv_2_2 = tf.nn.relu(deconv2d(deconv_2_1, W_deconv2_2, output_shape=[FLAGS.batch_size,160,160,64], padding='SAME'))
        tf.summary.histogram("W_deconv2_1", W_deconv2_1)
        tf.summary.histogram("W_deconv2_2", W_deconv2_2)
        tf.summary.image("channel1", tf.slice(deconv_2_2, [0,0,0,0],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(deconv_2_2, [0,0,0,1],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(deconv_2_2, [0,0,0,2],[-1,160,160,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(deconv_2_2, [0,0,0,3],[-1,160,160,1]), max_outputs=3)

    with tf.name_scope("unpool_1"):
        # shape: 320X320X64
        unpool_1 = unpooling(deconv_2_2, indices_1, shape=[FLAGS.batch_size,320,320,64])
        tf.summary.image("channel1", tf.slice(unpool_1, [0,0,0,0],[-1,320,320,1]), max_outputs=3)
        tf.summary.image("channel2", tf.slice(unpool_1, [0,0,0,1],[-1,320,320,1]), max_outputs=3)
        tf.summary.image("channel3", tf.slice(unpool_1, [0,0,0,2],[-1,320,320,1]), max_outputs=3)
        tf.summary.image("channel4", tf.slice(unpool_1, [0,0,0,3],[-1,320,320,1]), max_outputs=3)

    with tf.name_scope("deconv_1"):
        # shape: 320X320X1
        W_deconv1_1 = weight([3, 3, 64, 64], "weights_1")
        W_deconv1_2 = weight([3, 3, 1, 64], "weights_2")
        deconv_1_1 = tf.nn.relu(deconv2d(unpool_1, W_deconv1_1, output_shape=[FLAGS.batch_size,320,320,64], padding='SAME'))
        deconv_1_2 = tf.nn.relu(deconv2d(deconv_1_1, W_deconv1_2, output_shape=[FLAGS.batch_size,320,320,1], padding='SAME'))
        tf.summary.histogram("W_deconv1_1", W_deconv1_1)
        tf.summary.histogram("W_deconv1_2", W_deconv1_2)
        tf.summary.image("output_feature", deconv_1_2, max_outputs=3)

    with tf.name_scope("final_result"):
        output = tf.nn.sigmoid(deconv_1_2)
        result = 255 * tf.cast(output + 0.5, tf.uint8)
        tf.summary.image("sigmoid_out", output, max_outputs=3)
        tf.summary.image("segmentation", result, max_outputs=3)

    with tf.name_scope("evaluation"):
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = fg_gt, logits = deconv_1_2))
        tf.summary.scalar("loss", cross_entropy)

    with tf.name_scope('training_op'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_step = optimizer.minimize(cross_entropy)

    train_file = "train.tfrecords"
    test_file  = "test.tfrecords"
    saver = tf.train.Saver()
    img_size = [FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth]
    train_batch = tf.train.shuffle_batch([read_tfrecord(train_file, img_size)],
                batch_size = FLAGS.batch_size,
                capacity = 3000,
                num_threads = 2,
                min_after_dequeue = 1000)
    test_batch = tf.train.shuffle_batch([read_tfrecord(test_file, img_size)],
                batch_size = FLAGS.batch_size,
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
                train_step.run({frame_and_bg:inputs_train, fg_gt:outputs_gt_train, learning_rate:1e-3})
            elif iter <= FLAGS.max_iteration - 1000:
                train_step.run({frame_and_bg:inputs_train, fg_gt:outputs_gt_train, learning_rate:0.5e-3})
            else:
                train_step.run({frame_and_bg:inputs_train, fg_gt:outputs_gt_train, learning_rate:1e-4})
            # print training loss and test loss
            if iter%10 == 0:
                summary_train = sess.run(summary, {frame_and_bg:inputs_train, fg_gt:outputs_gt_train})
                train_writer.add_summary(summary_train, iter)
                train_writer.flush()
                summary_test = sess.run(summary, {frame_and_bg:inputs_test, fg_gt:outputs_gt_test})
                test_writer.add_summary(summary_test, iter)
                test_writer.flush()
            # record training loss and test loss
            if iter%10 == 0:
                train_loss  = cross_entropy.eval({frame_and_bg:inputs_train, fg_gt:outputs_gt_train})
                test_loss   = cross_entropy.eval({frame_and_bg:inputs_test, fg_gt:outputs_gt_test})
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
            final_test = final_test + cross_entropy.eval({frame_and_bg:inputs_test, fg_gt:outputs_gt_test})
        final_test = final_test / 5.
        print("final test loss %f" % final_test)
        coord.request_stop()
        coord.join(threads)

        running_time = time.time() - start_time
        hour = int(running_time / 3600)
        minute = int((running_time % 3600) / 60)
        second = (running_time % 3600) % 60
        print("running time: %d h %d min %d sec" % (hour, minute, second))
