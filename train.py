import numpy as np
import tensorflow as tf
from bgsCNN_v1 import bgsCNN_v1
from bgsCNN_v2 import bgsCNN_v2
from bgsCNN_v3 import bgsCNN_v3
from bgsCNN_v4 import bgsCNN_v4
from bgsCNN_v5 import bgsCNN_v5
from generate_bg import generate_bg
from prepare_data import prepare_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("generate_bg", False, "whether to run generate_bg()")
tf.app.flags.DEFINE_boolean("prepare_data", True, "whether to run prepare_data()")
tf.app.flags.DEFINE_string("dataset_dir", "", "directory of dataset (with background images)")

tf.app.flags.DEFINE_string("train_file", "", "tfrecords file for training data")
tf.app.flags.DEFINE_string("test_file", "", "tfrecords file for test data")
tf.app.flags.DEFINE_string("log_dir", "", "directory of recording training logs")

tf.app.flags.DEFINE_integer("model_version", 2, "version number of the model; default as the best model(v2)")
tf.app.flags.DEFINE_integer("train_batch_size", 40, "size of training batch")
tf.app.flags.DEFINE_integer("test_batch_size", 200, "size of test batch")
tf.app.flags.DEFINE_integer("max_iteration", 10000, "maximum # of training steps")
tf.app.flags.DEFINE_integer("image_height", 321, "height of inputs")
tf.app.flags.DEFINE_integer("image_width", 321, "width of inputs")
tf.app.flags.DEFINE_integer("image_depth", 7, "depth of inputs")

def main(_):
    # check FLAGS
    if FLAGS.generate_bg:
        generate_bg(FLAGS.dataset_dir)
    if FLAGS.prepare_data:
        prepare_data(FLAGS.dataset_dir, FLAGS.image_height, FLAGS.image_width)
        FLAGS.train_file = "train.tfrecords"
        FLAGS.test_file = "test.tfrecords"
    if (not FLAGS.prepare_data) & (FLAGS.train_file == ""):
        print("please specify the tfrecords file for training data")
        return
    if (not FLAGS.prepare_data) & (FLAGS.test_file == ""):
        print("please specify the tfrecords file for test data")
        return
    if FLAGS.log_dir == "":
        print("please specify the directory of recording training logs")
        return
    # build model
    if FLAGS.model_version == 1:
        model = bgsCNN_v1(train_file=FLAGS.train_file, test_file=FLAGS.test_file, log_dir=FLAGS.log_dir,
                        train_batch_size=FLAGS.train_batch_size, test_batch_size=FLAGS.test_batch_size,
                        max_iteration=FLAGS.max_iteration,
                        image_height=FLAGS.image_height, image_width=FLAGS.image_width, image_depth=FLAGS.image_depth)
    elif FLAGS.model_version == 2:
        model = bgsCNN_v2(train_file=FLAGS.train_file, test_file=FLAGS.test_file, log_dir=FLAGS.log_dir,
                        train_batch_size=FLAGS.train_batch_size, test_batch_size=FLAGS.test_batch_size,
                        max_iteration=FLAGS.max_iteration,
                        image_height=FLAGS.image_height, image_width=FLAGS.image_width, image_depth=FLAGS.image_depth)
    elif FLAGS.model_version == 3:
        model = bgsCNN_v3(train_file=FLAGS.train_file, test_file=FLAGS.test_file, log_dir=FLAGS.log_dir,
                        train_batch_size=FLAGS.train_batch_size, test_batch_size=FLAGS.test_batch_size,
                        max_iteration=FLAGS.max_iteration,
                        image_height=FLAGS.image_height, image_width=FLAGS.image_width, image_depth=FLAGS.image_depth)
    elif FLAGS.model_version == 4:
        model = bgsCNN_v4(train_file=FLAGS.train_file, test_file=FLAGS.test_file, log_dir=FLAGS.log_dir,
                        train_batch_size=FLAGS.train_batch_size, test_batch_size=FLAGS.test_batch_size,
                        max_iteration=FLAGS.max_iteration,
                        image_height=FLAGS.image_height, image_width=FLAGS.image_width, image_depth=FLAGS.image_depth)
    elif FLAGS.model_version == 5:
        model = bgsCNN_v5(train_file=FLAGS.train_file, test_file=FLAGS.test_file, log_dir=FLAGS.log_dir,
                        train_batch_size=FLAGS.train_batch_size, test_batch_size=FLAGS.test_batch_size,
                        max_iteration=FLAGS.max_iteration,
                        image_height=FLAGS.image_height, image_width=FLAGS.image_width, image_depth=FLAGS.image_depth)
    else:
        print("The model version is not supported. Please choose from 1 to 5")
    # run training
    model.train()

if __name__ == '__main__':
    tf.app.run()
