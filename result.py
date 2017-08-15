import cv2
import numpy as np
import libbgs
import os
import os.path
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim

def walklevel(some_dir, level):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

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

def inference(input):
    '''
    input:
    a placeholder of shape [None, FLAGS.image_height, FLAGS.image_height, 6]
    '''
    with tf.name_scope("pre_conv"):
        # shape: 321X321X3
        W_pre = weight([1, 1, 6, 3], "weights")
        pre_conv = conv2d(frame_and_bg, W_pre)
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
            output_shape = [1, 43, 43, 32], strides = [1, 2, 2, 1])
    with tf.name_scope("deconv_1_max_pooling"):
        # shape: 41X41X16
        deconv_1_pool = pool3d(deconv_1, [1,2,3,3,1], [1,2,1,1,1], 'max')
    with tf.name_scope("deconv_2"):
        # shape: 83X83X8
        W_deconv2 = weight([3, 3, 8, 16], "weights")
        deconv_2 = deconv2d(deconv_1_pool, W_deconv2,
            output_shape = [1, 83, 83, 8], strides = [1, 2, 2, 1])
    with tf.name_scope("deconv_2_max_pooling"):
        # shape: 81X81X8
        deconv_2_pool = tf.nn.max_pool(deconv_2, [1,3,3,1], [1,1,1,1], 'VALID')
    with tf.name_scope("deconv_3"):
        # shape: 163X163X4
        W_deconv3 = weight([3, 3, 4, 8], "weights")
        deconv_3 = deconv2d(deconv_2_pool, W_deconv3,
            output_shape = [1, 163, 163, 4], strides = [1, 2, 2, 1])
    with tf.name_scope("deconv_3_max_pooling"):
        # shape: 161X161X4
        deconv_3_pool = tf.nn.max_pool(deconv_3, [1,3,3,1], [1,1,1,1], 'VALID')
    with tf.name_scope("deconv_4"):
        # shape: 323X323X1
        W_deconv4 = weight([3, 3, 1, 4], "weights")
        deconv_4 = deconv2d(deconv_3_pool, W_deconv4,
            output_shape = [1, 323, 323, 1], strides = [1, 2, 2, 1])
    with tf.name_scope("deconv_4_max_pooling"):
        # shape: 321X321X1
        deconv_4_pool = tf.nn.max_pool(deconv_4, [1,3,3,1], [1,1,1,1], 'VALID')
    with tf.name_scope("conv"):
        W_conv = weight([1, 1, 1, 1], "weights")
        conv = conv2d(deconv_4_pool, W_conv)
    with tf.name_scope("final_result"):
        output = tf.nn.sigmoid(conv)
    return output

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("image_height", 321, "height of inputs")
    tf.app.flags.DEFINE_integer("image_width", 321, "width of inputs")
    tf.app.flags.DEFINE_integer("image_depth", 7, "depth of inputs")
    # the inference model
    frame_and_bg = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_height, 6])
    out = inference(frame_and_bg)
    # background subtractor
    bgs = libbgs.SuBSENSE()

    cv2.namedWindow("frame")
    cv2.namedWindow("foreground mask")
    img_size = [FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "logs/model.ckpt-9600")
        for __, dirnames_l1, __ in walklevel("dataset2014", level = 1):
            for dirname_l1 in dirnames_l1:
                if (dirname_l1 != "dataset") & (dirname_l1 != "results"):
                    for __, dirnames_l2, __ in walklevel("dataset2014/dataset/" + dirname_l1, level = 0):
                        for dirname_l2 in dirnames_l2:
                            src_dir = "dataset2014/dataset/" + dirname_l1 + "/" + dirname_l2
                            result_dir = "dataset2014/results/" + dirname_l1 + "/" + dirname_l2
                            F = open(src_dir + "/temporalROI.txt", 'r')
                            line  = F.read().split(' ')
                            end = int(line[1])
                            num = 1
                            ROI = cv2.imread(src_dir + "/ROI.bmp")
                            original_size = ROI.shape; original_size[2] = 1
                            while num <= end:
                                # get the background model
                                frame_file = src_dir + "/input/" + num2filename(num, "in") + ".jpg"
                                frame = cv2.imread(frame_file)
                                check = (frame[:,:,0] == frame[:,:,1])
                                if check.all():
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                __ = bgs.apply(frame)
                                bg_model = bgs.getBackgroundModel()
                                # build the input data cube for CNN
                                frame = cv2.imread(frame_file)
                                frame[ROI == 0] = 0
                                frame = cv2.resize(frame, (FLAGS.image_width, FLAGS.image_height),
                                                    interpolation = cv2.INTER_CUBIC)
                                bg_model[ROI == 0] = 0
                                bg_model = cv2.resize(bg_model, (FLAGS.image_width, FLAGS.image_height),
                                                    interpolation = cv2.INTER_CUBIC)
                                data_cube = np.uint8(np.concatenate([frame, bg_model], 2))
                                # feed forward the CNN
                                CNN_out = sess.run(out, {frame_and_bg:data_cube})
                                result = cv2.resize(CNN_out, original_size, interpolation = cv2.INTER_CUBIC)
                                # record the result
                                cv2.imwrite(result_dir + "/" + num2filename(num, "bin") + ".png")
                                # show
                                cv2.imshow("frame", cv2.imread(frame_file))
                                cv2.imshow("foreground mask", result)
                                num = num + 1
