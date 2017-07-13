import tensorflow as tf
import cv2
import numpy as np
import os
import os.path

from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import slim

img = np.float32(cv2.imread("1.jpg"))
img = cv2.normalize(img, 0., 1. ,cv2.NORM_MINMAX)
inputs = np.ones([1,240, 320, 3], dtype = np.float32)
inputs[0,:,:,:] = img

print slim.get_model_variables("CNN_models/resnet_v1_152")
# init_fn = slim.assign_from_checkpoint_fn(
#         os.path.join("CNN_models", "resnet_v1_152.ckpt"))
# with slim.arg_scope(resnet_v1.resnet_arg_scope()):
#     net, end_points = resnet_v1.resnet_v1_152(
#         inputs,
#         num_classes = None,
#         is_training = True,
#         global_pool = False,
#         output_stride = None,)
# init = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     # init_fn(sess)
#     n = sess.run(net)
#     print n

# filename_queue = tf.train.string_input_producer(["train.tfrecords"])
# reader = tf.TFRecordReader()
# __, serialized_example = reader.read(filename_queue)
# feature={
#       'image_raw': tf.FixedLenFeature([], tf.string),
#       'height': tf.FixedLenFeature([], tf.int64),
#       'width': tf.FixedLenFeature([], tf.int64),
#       'depth': tf.FixedLenFeature([], tf.int64),
# }
# features = tf.parse_single_example(serialized_example, features=feature)
# image = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [240,320,9])
#
# batch = tf.train.shuffle_batch([image],
#             batch_size = 30,
#             capacity = 1000 + 3 * 30,
#             num_threads = 2,
#             min_after_dequeue = 1000)
#
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     img = sess.run(image)
#     print type(img)
#     print img.shape
#
#     coord.request_stop()
#     coord.join(threads)
