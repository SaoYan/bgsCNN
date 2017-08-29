import cv2
import os
import os.path
import numpy as np
import tensorflow as tf

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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def prepare_data(root_dir, height, width):
    total_num_train = 0; total_num_test = 0
    train_writer = tf.python_io.TFRecordWriter("train.tfrecords")
    test_writer  = tf.python_io.TFRecordWriter("test.tfrecords")
    for __, dirnames_l0, __ in walklevel(root_dir, level = 0):
        for dirname_l0 in dirnames_l0:
            print ("start dealing with " + dirname_l0)
            F = open(root_dir + "/" + dirname_l0 + "/temporalROI.txt", 'r')
            line  = F.read().split(' ')
            begin = int(line[0]); end = int(line[1])
            num = begin
            roi_filename = root_dir + "/" + dirname_l0 + "/ROI.bmp"
            roi_img = cv2.imread(roi_filename)
            while num <= end:
                frame_filename = root_dir + "/" + dirname_l0 + "/input/" + num2filename(num, "in") + ".jpg"
                bg_filename = root_dir + "/" + dirname_l0 + "/bg/" + num2filename(num, "bg") + ".jpg"
                gt_filename = root_dir + "/" + dirname_l0 + "/groundtruth/" + num2filename(num, "gt") + ".png"
                frame = cv2.imread(frame_filename)
                frame[roi_img == 0] = 0
                frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_CUBIC)
                bg_model = cv2.imread(bg_filename)
                bg_model[roi_img == 0] = 0
                bg_model = cv2.resize(bg_model, (width, height), interpolation = cv2.INTER_CUBIC)
                gt_mask = cv2.imread(gt_filename)
                gt_mask[(gt_mask != 0) & (gt_mask != 255)] = 0
                gt_mask = cv2.resize(gt_mask, (width, height))
                gt_mask[gt_mask > 0] = 255

                gt = gt_mask[:,:,0]
                gt = np.expand_dims(gt, axis = 2)
                data_cube = np.uint8(np.concatenate([frame, bg_model, gt], 2))
                image_raw = data_cube.tostring()
                feature={ 'image_raw': _bytes_feature(image_raw) }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                if dirname_l0 != "highway":
                    train_writer.write(example.SerializeToString())
                    total_num_train = total_num_train + 1
                else:
                    test_writer.write(example.SerializeToString())
                    total_num_test = total_num_test + 1
                print("add data for frame # " + str(num))
                num = num + 1
            print ("finish dealing with " + dirname_l0 + "\n")
    print ("total # of training samples: " + str(total_num_train))
    print ("total # of test samples: " + str(total_num_test))
