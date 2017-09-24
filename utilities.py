import os
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

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

def vgg_16(inputs,
           variables_collections=None,
           scope='vgg_16',
           reuse=None):
    """
    modification of vgg_16 in TF-slim
    see original code in https://github.com/tensorflow/models/blob/master/slim/nets/vgg.py
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            conv1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', biases_initializer=None,
                            variables_collections=variables_collections, reuse=reuse)
            pool1, argmax_1 = tf.nn.max_pool_with_argmax(conv1, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool1')
            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2', biases_initializer=None,
                            variables_collections=variables_collections, reuse=reuse)
            pool2, argmax_2 = tf.nn.max_pool_with_argmax(conv2, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool2')
            conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3', biases_initializer=None,
                            variables_collections=variables_collections, reuse=reuse)
            pool3, argmax_3 = tf.nn.max_pool_with_argmax(conv3, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool3')
            conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4', biases_initializer=None,
                            variables_collections=variables_collections, reuse=reuse)
            pool4, argmax_4 = tf.nn.max_pool_with_argmax(conv4, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool4')
            conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5', biases_initializer=None,
                            variables_collections=variables_collections, reuse=reuse)
            pool5, argmax_5 = tf.nn.max_pool_with_argmax(conv5, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool5')
            # return argmax
            argmax = (argmax_1, argmax_2, argmax_3, argmax_4, argmax_5)
            # return feature maps
            features = (conv1, conv2, conv3, conv4, conv5)
            return pool5, argmax, features

def unpool(pool, ind, shape, ksize=[1, 2, 2, 1], scope=None):
    with tf.name_scope(scope):
        input_shape =  tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]
        flat_input_size = tf.cumprod(input_shape)[-1]
        flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])
        pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                shape=tf.stack([input_shape[0], 1, 1, 1]))
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, tf.stack([flat_input_size, 1]))
        ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, tf.stack(output_shape))
        ret = tf.reshape(ret, shape=shape)
        return ret

def upsample(value, scope=None):
    with tf.name_scope(scope):
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = tf.reshape(value, [-1] + sh[-dim:])
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size)
        return out

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
    input_cast = img_batch[:,:,:,0:6].astype(dtype = np.float32)
    input_min = np.amin(input_cast, axis=(1,2,3))
    input_max = np.amax(input_cast, axis=(1,2,3))
    for i in range(3):
        input_min = np.expand_dims(input_min, i+1)
        input_max = np.expand_dims(input_max, i+1)
    input_norm = (input_cast - input_min) / (input_max - input_min)
    gt_cast = img_batch[:,:,:,6].astype(dtype = np.float32)
    gt_cast = np.expand_dims(gt_cast, 3)
    gt_min = np.amin(gt_cast, axis=(1,2,3))
    gt_max = np.amax(gt_cast, axis=(1,2,3))
    for i in range(3):
        gt_min = np.expand_dims(gt_min, i+1)
        gt_max = np.expand_dims(gt_max, i+1)
    gt_norm = (gt_cast - gt_min) / (gt_max - gt_min)
    return input_norm, gt_norm

def walklevel(some_dir, level):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def num2filename(num, prefix):
    if num < 10:
        return prefix + "00000" + str(num)
    elif num < 100:
        return prefix + "0000" + str(num)
    elif num < 1000:
        return prefix + "000" + str(num)
    elif num < 10000:
        return prefix + "00" + str(num)
    elif num < 100000:
        return prefix + "0" + str(num)
    else:
        return prefix + str(num)
