#! /usr/bin/python

from generate_bg import generate_bg
from prepare_data import prepare_data

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim

# generate_bg()
prepare_data(321, 321)

# def weight(shape, name):
# 	initial = tf.random_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
# 	return tf.Variable(initial,name=name)
#
# def bias(shape, name):
# 	initial = tf.constant(0, shape=shape, dtype=tf.float32)
# 	return tf.Variable(initial,name=name)
#
# def conv2d(x, W):
# 	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')
#
# if __name__ == '__main__':
#     image_height = 240; image_width = 320
#
# 	with tf.name_scope("input_data"):
#         frame_and_bg = tf,placeholder(tf.float32, [None, image_height, image_height, 6])
#         fg_gt = tf.placeholder(tf.float32, [None, ])
#
# 	with slim.arg_scope(resnet_v2.resnet_arg_scope()):
# 	    net, end_points = resnet_v2.resnet_v2_152(
# 	        inputs,
# 	        num_classes = None,
# 	        is_training = True,
# 	        global_pool = False)
