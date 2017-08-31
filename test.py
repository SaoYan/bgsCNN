import cv2
import numpy as np
import tensorflow as tf
from utilities import *
from bgsCNN_v1 import bgsCNN_v1
from bgsCNN_v2 import bgsCNN_v2
from bgsCNN_v3 import bgsCNN_v3
from bgsCNN_v4 import bgsCNN_v4

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("batch_size", 10, "size of training batch")
tf.app.flags.DEFINE_integer("image_height", 321, "height of inputs")
tf.app.flags.DEFINE_integer("image_width", 321, "width of inputs")
tf.app.flags.DEFINE_integer("image_depth", 7, "depth of inputs")
tf.app.flags.DEFINE_integer("optimal_step", None, "# iteration step corresponding to the minimum test loss")
tf.app.flags.DEFINE_integer("model_version", 2, "version number of the model; default as the best model(v2)")
tf.app.flags.DEFINE_string("test_file", "", "tfrecords file for test data")
tf.app.flags.DEFINE_string("log_dir", "", "directory of training logs")

def main(_):
    # check FLAGS
    if FLAGS.test_file == "":
        print("please specify the tfrecords file for test data")
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
    # test on the whole test set
    img_size = [FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth]
    saver = tf.train.Saver()
    test_batch = tf.train.shuffle_batch([read_tfrecord(FLAGS.test_file, img_size)],
                batch_size = FLAGS.batch_size,
                capacity = 10*FLAGS.batch_size,
                num_threads = 2,
                min_after_dequeue = 5*FLAGS.batch_size)
    loss = 0.
    with tf.Session() as sess:
        test_writer  = tf.summary.FileWriter(FLAGS.log_dir + "/model_test", sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, FLAGS.log_dir + "/model.ckpt-" + str(FLAGS.optimal_step))
        for i in range(500):
            inputs_test, outputs_gt_test = build_img_pair(sess.run(test_batch))
            summary_test = sess.run(model.summary, {model.input_data:inputs_test, model.gt:outputs_gt_test, model.batch_size:FLAGS.batch_size})
            test_writer.add_summary(summary_test, i)
            l = model.cross_entropy.eval({model.input_data:inputs_test, model.gt:outputs_gt_test, model.batch_size:FLAGS.batch_size})
            loss = loss + l
            print("test loss %d: %f" % (i+1, l))
        print("average loss on test set: %f" % (loss/500.))
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
