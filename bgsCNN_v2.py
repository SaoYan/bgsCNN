from utilities import *
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim

class bgsCNN_v2:
    def __init__(self, train_file, test_file, log_dir,
                 train_batch_size = 40, test_batch_size  = 200,
                 max_iteration = 10000,
                 image_height = 321, image_width = 321, image_depth = 7):
        self.train_file = train_file
        self.test_file  = test_file
        self.log_dir = log_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_iteration = max_iteration
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.session = tf.Session()
        self.build_inputs()
        self.build_model()
        self.build_loss()
        self.build_optimizer()
        self.build_summary()

    def build_inputs(self):
        with tf.name_scope("input_data"):
            self.input_data = tf.placeholder(tf.float32, [None, self.image_height, self.image_height, 6])
            self.gt = tf.placeholder(tf.float32, [None, self.image_height, self.image_height, 1])
            self.learning_rate = tf.placeholder(tf.float32, [])
            self.batch_size = tf.placeholder(tf.int32, [])
            frame = tf.slice(self.input_data, [0,0,0,0], [-1,self.image_height, self.image_height, 3])
            bg = tf.slice(self.input_data, [0,0,0,3], [-1,self.image_height, self.image_height, 3])
            tf.summary.image("frame", frame, max_outputs=3)
            tf.summary.image("background", bg, max_outputs=3)
            tf.summary.image("groundtruth", self.gt, max_outputs=3)

    def build_model(self):
        with tf.name_scope("pre_conv"):
            # shape: 321X321X3
            W_pre = weight([1, 1, 6, 3], "weights")
            pre_conv = conv2d(self.input_data, W_pre)
            tf.summary.histogram("W_pre_conv", W_pre)
            tf.summary.image("pre_conv_out", pre_conv, max_outputs=3)
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
                output_shape = [self.batch_size, 43, 43, 32], strides = [1, 2, 2, 1])
            tf.summary.histogram("W_deconv1", W_deconv1)
            tf.summary.image("channel1", tf.slice(deconv_1, [0,0,0,0],[-1,43,43,1]), max_outputs=3)
            tf.summary.image("channel2", tf.slice(deconv_1, [0,0,0,1],[-1,43,43,1]), max_outputs=3)
            tf.summary.image("channel3", tf.slice(deconv_1, [0,0,0,2],[-1,43,43,1]), max_outputs=3)
            tf.summary.image("channel4", tf.slice(deconv_1, [0,0,0,3],[-1,43,43,1]), max_outputs=3)
        with tf.name_scope("deconv_1_max_pooling"):
            # shape: 41X41X16
            deconv_1_pool = pool3d(deconv_1, [1,2,3,3,1], [1,2,1,1,1], 'max')
            tf.summary.image("channel1", tf.slice(deconv_1_pool, [0,0,0,0],[-1,41,41,1]), max_outputs=3)
            tf.summary.image("channel2", tf.slice(deconv_1_pool, [0,0,0,1],[-1,41,41,1]), max_outputs=3)
            tf.summary.image("channel3", tf.slice(deconv_1_pool, [0,0,0,2],[-1,41,41,1]), max_outputs=3)
            tf.summary.image("channel4", tf.slice(deconv_1_pool, [0,0,0,3],[-1,41,41,1]), max_outputs=3)
        with tf.name_scope("deconv_2"):
            # shape: 83X83X8
            W_deconv2 = weight([3, 3, 8, 16], "weights")
            deconv_2 = deconv2d(deconv_1_pool, W_deconv2,
                output_shape = [self.batch_size, 83, 83, 8], strides = [1, 2, 2, 1])
            tf.summary.histogram("W_deconv2", W_deconv2)
            tf.summary.image("channel1", tf.slice(deconv_2, [0,0,0,0],[-1,83,83,1]), max_outputs=3)
            tf.summary.image("channel2", tf.slice(deconv_2, [0,0,0,1],[-1,83,83,1]), max_outputs=3)
            tf.summary.image("channel3", tf.slice(deconv_2, [0,0,0,2],[-1,83,83,1]), max_outputs=3)
            tf.summary.image("channel4", tf.slice(deconv_2, [0,0,0,3],[-1,83,83,1]), max_outputs=3)
        with tf.name_scope("deconv_2_max_pooling"):
            # shape: 81X81X8
            deconv_2_pool = tf.nn.max_pool(deconv_2, [1,3,3,1], [1,1,1,1], 'VALID')
            tf.summary.image("channel1", tf.slice(deconv_2_pool, [0,0,0,0],[-1,81,81,1]), max_outputs=3)
            tf.summary.image("channel2", tf.slice(deconv_2_pool, [0,0,0,1],[-1,81,81,1]), max_outputs=3)
            tf.summary.image("channel3", tf.slice(deconv_2_pool, [0,0,0,2],[-1,81,81,1]), max_outputs=3)
            tf.summary.image("channel4", tf.slice(deconv_2_pool, [0,0,0,3],[-1,81,81,1]), max_outputs=3)
        with tf.name_scope("deconv_3"):
            # shape: 163X163X4
            W_deconv3 = weight([3, 3, 4, 8], "weights")
            deconv_3 = deconv2d(deconv_2_pool, W_deconv3,
                output_shape = [self.batch_size, 163, 163, 4], strides = [1, 2, 2, 1])
            tf.summary.histogram("W_deconv3", W_deconv3)
            tf.summary.image("channel1", tf.slice(deconv_3, [0,0,0,0],[-1,163,163,1]), max_outputs=3)
            tf.summary.image("channel2", tf.slice(deconv_3, [0,0,0,1],[-1,163,163,1]), max_outputs=3)
            tf.summary.image("channel3", tf.slice(deconv_3, [0,0,0,2],[-1,163,163,1]), max_outputs=3)
            tf.summary.image("channel4", tf.slice(deconv_3, [0,0,0,3],[-1,163,163,1]), max_outputs=3)
        with tf.name_scope("deconv_3_max_pooling"):
            # shape: 161X161X4
            deconv_3_pool = tf.nn.max_pool(deconv_3, [1,3,3,1], [1,1,1,1], 'VALID')
            tf.summary.image("channel1", tf.slice(deconv_3_pool, [0,0,0,0],[-1,161,161,1]), max_outputs=3)
            tf.summary.image("channel2", tf.slice(deconv_3_pool, [0,0,0,1],[-1,161,161,1]), max_outputs=3)
            tf.summary.image("channel3", tf.slice(deconv_3_pool, [0,0,0,2],[-1,161,161,1]), max_outputs=3)
            tf.summary.image("channel4", tf.slice(deconv_3_pool, [0,0,0,3],[-1,161,161,1]), max_outputs=3)
        with tf.name_scope("deconv_4"):
            # shape: 323X323X1
            W_deconv4 = weight([3, 3, 1, 4], "weights")
            deconv_4 = deconv2d(deconv_3_pool, W_deconv4,
                output_shape = [self.batch_size, 323, 323, 1], strides = [1, 2, 2, 1])
            tf.summary.histogram("W_deconv4", W_deconv4)
            tf.summary.image("out", tf.slice(deconv_4, [0,0,0,0],[-1,323,323,1]), max_outputs=3)
        with tf.name_scope("deconv_4_max_pooling"):
            # shape: 321X321X1
            deconv_4_pool = tf.nn.max_pool(deconv_4, [1,3,3,1], [1,1,1,1], 'VALID')
            tf.summary.image("out", tf.slice(deconv_4_pool, [0,0,0,0],[-1,321,321,1]), max_outputs=3)
        with tf.name_scope("conv"):
            W_conv = weight([1, 1, 1, 1], "weights")
            conv = conv2d(deconv_4_pool, W_conv)
            tf.summary.histogram("W_conv1", W_conv)
            tf.summary.image("out", conv, max_outputs=3)
        with tf.name_scope("final_result"):
            output = tf.nn.sigmoid(conv)
            result = 255 * tf.cast(output + 0.5, tf.uint8)
            tf.summary.image("sigmoid_out", output, max_outputs=3)
            tf.summary.image("segmentation", result, max_outputs=3)
        self.logits = conv
        self.sigmoid_out = output

    def build_loss(self):
        with tf.name_scope("evaluation"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.gt, logits = self.logits))
            tf.summary.scalar("loss", self.cross_entropy)

    def build_optimizer(self):
        with tf.name_scope('training_op'):
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.train_step = optimizer.minimize(self.cross_entropy)

    def build_summary(self):
        self.summary = tf.summary.merge_all()

    def train(self):
        img_size = [self.image_height, self.image_width, self.image_depth]
        train_batch = tf.train.shuffle_batch([read_tfrecord(self.train_file, img_size)],
                    batch_size = self.train_batch_size,
                    capacity = 3000,
                    num_threads = 2,
                    min_after_dequeue = 1000)
        test_batch = tf.train.shuffle_batch([read_tfrecord(self.test_file, img_size)],
                    batch_size = self.test_batch_size,
                    capacity = 500,
                    num_threads = 2,
                    min_after_dequeue = 300)
        init = tf.global_variables_initializer()
        init_fn = slim.assign_from_checkpoint_fn("resnet_v2_50.ckpt", slim.get_model_variables('resnet_v2'))
        saver = tf.train.Saver()
        with self.session as sess:
            sess.run(init)
            init_fn(sess)
            train_writer = tf.summary.FileWriter(self.log_dir + "/train", sess.graph)
            test_writer  = tf.summary.FileWriter(self.log_dir + "/test", sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            inputs_test, outputs_gt_test = build_img_pair(sess.run(test_batch))
            for iter in range(self.max_iteration):
                inputs_train, outputs_gt_train = build_img_pair(sess.run(train_batch))
                # train with dynamic learning rate
                if iter <= 500:
                    self.train_step.run({self.input_data:inputs_train, self.gt:outputs_gt_train,
                                    self.learning_rate:1e-3, self.batch_size:self.train_batch_size})
                elif iter <= self.max_iteration - 1000:
                    self.train_step.run({self.input_data:inputs_train, self.gt:outputs_gt_train,
                                    self.learning_rate:0.5e-3, self.batch_size:self.train_batch_size})
                else:
                    self.train_step.run({self.input_data:inputs_train, self.gt:outputs_gt_train,
                                    self.learning_rate:1e-4, self.batch_size:self.train_batch_size})
                # print training loss and test loss
                if iter%10 == 0:
                    summary_train = sess.run(self.summary, {self.input_data:inputs_train, self.gt:outputs_gt_train,
                                             self.batch_size:self.train_batch_size})
                    train_writer.add_summary(summary_train, iter)
                    train_writer.flush()
                    summary_test = sess.run(self.summary, {self.input_data:inputs_test, self.gt:outputs_gt_test,
                                             self.batch_size:self.test_batch_size})
                    test_writer.add_summary(summary_test, iter)
                    test_writer.flush()
                # record training loss and test loss
                if iter%10 == 0:
                    train_loss  = self.cross_entropy.eval({self.input_data:inputs_train, self.gt:outputs_gt_train,
                                                    self.batch_size:self.train_batch_size})
                    test_loss   = self.cross_entropy.eval({self.input_data:inputs_test, self.gt:outputs_gt_test,
                                                    self.batch_size:self.test_batch_size})
                    print("iter step %d trainning batch loss %f"%(iter, train_loss))
                    print("iter step %d test loss %f\n"%(iter, test_loss))
                # record model
                if iter%100 == 0:
                    saver.save(sess, self.log_dir + "/model.ckpt", global_step=iter)
            coord.request_stop()
            coord.join(threads)
