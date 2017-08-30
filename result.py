import cv2
import numpy as np
import tensorflow as tf
import libbgs
import os
import os.path
from utilities import *
from bgsCNN_v1 import bgsCNN_v1
from bgsCNN_v2 import bgsCNN_v2
from bgsCNN_v3 import bgsCNN_v3
from bgsCNN_v4 import bgsCNN_v4

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("image_height", 321, "height of inputs")
tf.app.flags.DEFINE_integer("image_width", 321, "width of inputs")
tf.app.flags.DEFINE_integer("optimal_step", None, "# iteration step corresponding to the minimum test loss")
tf.app.flags.DEFINE_integer("model_version", 2, "version number of the model; default as the best model(v2)")
tf.app.flags.DEFINE_string("dataset_dir", "dataset2014", "directory of the original CDnet 2014 dataset")
tf.app.flags.DEFINE_string("log_dir", "", "directory of training logs")

def main(_):
    # check FLAGS
    if FLAGS.dataset_dir == "":
        print("please specify the directory of the dataset")
        return
    if FLAGS.log_dir == "":
        print("please specify the directory of training logs")
        return
    if FLAGS.optimal_step == None:
        print("please specify the iteration step corresponding to the minimum test loss")
        return
    # the inference model
    if FLAGS.model_version == 1:
        model = bgsCNN_v1(image_height=FLAGS.image_height, image_width=FLAGS.image_width)
    elif FLAGS.model_version == 2:
        model = bgsCNN_v2(image_height=FLAGS.image_height, image_width=FLAGS.image_width)
    elif FLAGS.model_version == 3:
        model = bgsCNN_v3(image_height=FLAGS.image_height, image_width=FLAGS.image_width)
    elif FLAGS.model_version == 4:
        model = bgsCNN_v4(image_height=FLAGS.image_height, image_width=FLAGS.image_width)
    # generate results for the whole dataset
    cv2.namedWindow("frame")
    cv2.namedWindow("foreground mask")
    saver = tf.train.Saver()
    flag = True
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.log_dir + "/model.ckpt-" + str(FLAGS.optimal_step))
        for __, dirnames_l1, __ in walklevel(FLAGS.dataset_dir, level = 1):
            for dirname_l1 in dirnames_l1:
                if (dirname_l1 != "dataset") & (dirname_l1 != "results"):
                    for __, dirnames_l2, __ in walklevel(FLAGS.dataset_dir + "/dataset/" + dirname_l1, level = 0):
                        for dirname_l2 in dirnames_l2:
                            src_dir = FLAGS.dataset_dir + "/dataset/" + dirname_l1 + "/" + dirname_l2
                            result_dir = FLAGS.dataset_dir + "/results/" + dirname_l1 + "/" + dirname_l2
                            if not os.path.exists(src_dir + "/done"):
                                print("start processing " + dirname_l2)
                                bgs = libbgs.SuBSENSE()
                                F = open(src_dir + "/temporalROI.txt", 'r')
                                line  = F.read().split(' ')
                                end = int(line[1])
                                num = 1
                                ROI = cv2.imread(src_dir + "/ROI.bmp")
                                original_size = ROI.shape
                                while num <= end:
                                    # get the background model
                                    frame_file = src_dir + "/input/" + num2filename(num, "in") + ".jpg"
                                    frame = cv2.imread(frame_file)
                                    check = (frame[:,:,0] == frame[:,:,1])
                                    if check.all():
                                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    __ = bgs.apply(frame)
                                    bg_model = bgs.getBackgroundModel()
                                    if check.all():
                                        bg_model = np.expand_dims(bg_model, 2)
                                        bg_model = np.concatenate([bg_model,bg_model,bg_model], 2)
                                    # build the input data cube for CNN
                                    frame = cv2.imread(frame_file)
                                    frame[ROI == 0] = 0
                                    frame = cv2.resize(frame, (FLAGS.image_width, FLAGS.image_height),
                                                        interpolation = cv2.INTER_CUBIC)
                                    bg_model[ROI == 0] = 0
                                    bg_model = cv2.resize(bg_model, (FLAGS.image_width, FLAGS.image_height),
                                                        interpolation = cv2.INTER_CUBIC)
                                    data_cube = np.uint8(np.concatenate([frame, bg_model], 2))
                                    data_cube = np.expand_dims(data_cube, axis=0)
                                    # feed forward the CNN
                                    CNN_out = sess.run(model.sigmoid_out, {model.input_data:data_cube})
                                    CNN_out = np.squeeze(CNN_out, axis=0)
                                    CNN_out = cv2.medianBlur(CNN_out, 3)
                                    result = np.zeros(CNN_out.shape, dtype=np.uint8)
                                    result[CNN_out >= 0.5] = 255
                                    result = cv2.resize(result, (original_size[1], original_size[0]))
                                    result[result>=10] = 255
                                    result[result<10] = 0
                                    # record the result
                                    result_file = result_dir + "/" + num2filename(num, "bin") + ".png"
                                    cv2.imwrite(result_file, result)
                                    # show
                                    cv2.imshow("frame", cv2.imread(frame_file))
                                    cv2.imshow("foreground mask", result)
                                    num = num + 1
                                    # press 'q' to quit
                                    c = cv2.waitKey(30)
                                    if c >= 0:
                                        if chr(c) == 'q':
                                            flag = False
                                            break
                                        else:
                                            continue
                                    else:
                                        continue
                                if flag:
                                    print("finish processing " + dirname_l2 + "\n")
                                    os.makedirs(src_dir + "/done")

if __name__ == '__main__':
    tf.app.run()
