import cv2
import libbgs
import numpy as np
import tensorflow as tf
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
tf.app.flags.DEFINE_string("video_file", "", "video file for test")
tf.app.flags.DEFINE_string("log_dir", "", "directory of training logs")

def main(_):
    # check FLAGS
    if FLAGS.log_dir == "":
        print("please specify the directory of training logs")
        return
    if FLAGS.video_file == "":
        print("please specify the test video")
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
    # run on the video
    bgs = libbgs.SuBSENSE()
    saver = tf.train.Saver()
    cap = cv2.VideoCapture()
    cap.open(FLAGS.video_file)
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.log_dir + "/model.ckpt-" + str(FLAGS.optimal_step))
        while True:
            # current frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_shape = frame.shape
            frame_store = np.copy(frame)
            # generate background image
            check = (frame[:,:,0] == frame[:,:,1])
            if check.all():
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask_reference = bgs.apply(frame)
            bg_model = bgs.getBackgroundModel()
            # generate input date for CNN
            frame = cv2.resize(frame, (FLAGS.image_width,FLAGS.image_height), cv2.INTER_CUBIC)
            bg_model = cv2.resize(bg_model, (FLAGS.image_width,FLAGS.image_height), cv2.INTER_CUBIC)
            input_data = np.concatenate([frame, bg_model], 2)
            input_data = np.expand_dims(input_data, 0)
            input_cast = input_data.astype(dtype = np.float32)
            input_min = np.amin(input_cast)
            input_max = np.amax(input_cast)
            input_norm = (input_cast - input_min) / (input_max - input_min)
            # feed forward through CNN
            output_img = sess.run(model.sigmoid_out, {model.input_data:input_norm, model.batch_size:1})
            output_img = np.squeeze(output_img, axis=0)
            # post processing to get the final foregrond mask
            output_img = cv2.medianBlur(output_img, ksize = 3)
            mask = np.zeros(output_img.shape, dtype = np.uint8)
            mask[output_img >= 0.5] = 255
            mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
            mask[mask>=10] = 255
            mask[mask<10] = 0
            # show results
            cv2.namedWindow("frame"); cv2.imshow("frame",frame_store)
            cv2.namedWindow("mask"); cv2.imshow("mask",mask)
            cv2.namedWindow("reference mask"); cv2.imshow("reference mask", mask_reference)
            # press "q" to quitting
            c = cv2.waitKey(20)
            if c >= 0:
                if chr(c) == 'q':
                    break
                else:
                    continue
            else:
                continue

if __name__ == '__main__':
    tf.app.run()
