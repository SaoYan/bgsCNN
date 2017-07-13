#! /usr/bin/python

import tensorflow as tf
import cv2
import numpy as np
from generate_bg import generate_bg
from prepare_data import prepare_data

#generate_bg()
prepare_data()

def weight(shape, name):
	initial = tf.random_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def bias(shape, name):
	initial = tf.constant(0, shape=shape, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')

if __name__ == '__main__':
    image_height = 240; image_width = 320
    with tf.name_scope("input_data"):
        frame_and_bg = tf,placeholder(tf.float32, [None, image_height, image_height, 6])
        fg_gt = tf.placeholder(tf.float32, [None, ])
