#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import train_operation as op

MAX_STEPS = 10000000
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "train.csv"
COARSE_DIR = "coarse"
REFINE_DIR = "refine"

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)
        images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        logits = model.inference(images, keep_conv, keep_hidden)
        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        init_op = tf.global_variables_initializer()

        # Session
        with tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT)) as sess:
            sess.run(init_op)
            # parameters
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./train', sess.graph)
            test_writer = tf.summary.FileWriter('./test')
            tf.global_variables_initializer().run()
            
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" %(variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue

            # train
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for step in xrange(MAX_STEPS):
                index = 0
                for i in xrange(1000):
                    summary, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                    if index % 10 == 0:
                        #test_writer.add_summary(summary, index)
                        print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    if index % 500 == 0:
                        output_predict(logits_val, images_val, "data/predict_%05d_%05d" % (step, i))

                    #train_writer.add_summary(summary, index)
                    index += 1

            coord.request_stop()
            coord.join(threads)



def main(argv=None):
    if not gfile.Exists("./train"):
        gfile.MakeDirs("./train")
    if not gfile.Exists("./test"):
        gfile.MakeDirs("./test")
    train()


if __name__ == '__main__':
    tf.app.run()
