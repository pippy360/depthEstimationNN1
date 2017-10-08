# encoding: utf-8

# tensorflow
import tensorflow as tf
import math

TOWER_NAME = 'tower'
UPDATE_OPS_COLLECTION = '_update_ops_'


def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_gpu(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding)
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv_ = tf.nn.relu(bias, name=scope.name)
        return conv_


def fc(scope_name, inputs, shape, bias_shape, wd=0.04, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        flat = tf.reshape(inputs, [-1, shape[0]])
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
        fc = tf.nn.relu_layer(flat, weights, biases, name=scope.name)
        return fc


def inference(images, reuse=False, trainable=True):
    #input should be [n,240,320,3]
    wd = 0.0
    conv1 = conv2d('conv1', images, [7, 7, 3, 256], [256], [1, 16, 16, 1], padding='SAME', reuse=reuse, trainable=trainable)
    conv1 = tf.contrib.layers.batch_norm(conv1)
    multScalar = _variable_with_weight_decay(
            'weights1',
            shape=[1],
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
    addScalar = _variable_with_weight_decay(
            'weights2',
            shape=[1],
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
    conv1 = tf.multiply(conv1, multScalar)
    conv1 = tf.add(conv1, addScalar)
    conv1 = tf.nn.relu(conv1, 'relu')
    print "conv1"
    print conv1

    #connect these two layers

    coarse6 = fc('coarse6', conv1, [256, 4096], [4096], reuse=reuse, trainable=trainable)
    coarse7 = fc('coarse7', coarse6, [4096, 4070], [4070], reuse=reuse, trainable=trainable)
    coarse7_output = tf.reshape(coarse7, [-1, 55, 74, 1])
    print "coarse7_output"
    print coarse7_output
    return coarse7_output

def loss(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, 55*74])
    depths_flat = tf.reshape(depths, [-1, 55*74])
    print "logits_flat"
    print logits_flat
    print "depths_flat"
    print depths_flat
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 55*74])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / 55.0*74.0 - 0.5*sqare_sum_d / math.pow(55*74, 2))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
