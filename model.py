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


def conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, trainable=True):
    with tf.variable_scope(scope_name) as scope:
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

        print "inputs"
        print inputs
        print [-1, shape[0]]
        flat = tf.reshape(inputs, [-1, shape[0]])
        print "Creating fully connected layer with:"
        print flat
        print "#######"
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

def oneRun(scope_name, inputs, convOutputSize, kernelSize=3, stride=1):
    with tf.variable_scope(scope_name) as scope:
        wd = 0.0
        conv1 = conv2d('conv1', inputs, [kernelSize, kernelSize, 3, convOutputSize], [convOutputSize], [1, stride, stride, 1], padding='SAME', trainable=True)
        conv1 = tf.contrib.layers.batch_norm(conv1)
        multScalar = _variable_with_weight_decay(
                'multWeight',
                shape=[1],
                stddev=0.01,
                wd=wd,
                trainable=True
            )
        addScalar = _variable_with_weight_decay(
                'addWeight',
                shape=[1],
                stddev=0.01,
                wd=wd,
                trainable=True
            )

        conv1 = tf.multiply(conv1, multScalar)#try removing these and see what happens
        conv1 = tf.add(conv1, addScalar)#try removing these and see what happens
        conv1 = tf.nn.relu(conv1, 'relu')
        return conv1

def oneRunWithoutRelu(scope_name, inputs, convOutputSize, kernelSize=3, stride=1):
    with tf.variable_scope(scope_name) as scope:
        wd = 0.0
        conv1 = conv2d('conv1', inputs, [kernelSize, kernelSize, 3, convOutputSize], [convOutputSize], [1, stride, stride, 1], padding='SAME', trainable=True)
        conv1 = tf.contrib.layers.batch_norm(conv1)
        multScalar = _variable_with_weight_decay(
                'multWeight',
                shape=[1],
                stddev=0.01,
                wd=wd,
                trainable=True
            )
        addScalar = _variable_with_weight_decay(
                'addWeight',
                shape=[1],
                stddev=0.01,
                wd=wd,
                trainable=True
            )

        conv1 = tf.multiply(conv1, multScalar)#try removing these and see what happens
        conv1 = tf.add(conv1, addScalar)#try removing these and see what happens
        return conv1


def maxPool(scope_name, inputs, kernelSize, stride):
    with tf.variable_scope(scope_name) as scope:
        max1 = tf.nn.max_pool(inputs, [1, kernelSize, kernelSize, 1], [1, stride, stride, 1], padding='SAME')
        return max1

def inference(images, reuse=False, trainable=True):
    #input should be [n,240,320,3]

    #connect these two layers
    conv1 = oneRun("conv1", images, convOutputSize=64, kernelSize=7, stride=2)
    conv1b = maxPool("max1", conv1, kernelSize=3, stride=2)

    conv1 = oneRun("conv2", conv1b, convOutputSize=64,  kernelSize=1, stride=1)
    conv1 = oneRun("conv3", conv1, convOutputSize=64,  kernelSize=3, stride=1)
    conv1 = oneRunWithoutRelu("conv4", conv1, convOutputSize=256, kernelSize=1, stride=1)

    #resize the original input
    conv1b = oneRunWithoutRelu("conv5", conv1b, convOutputSize=256, kernelSize=1, stride=1)

    #concat
    conv1 = conv1 + conv1b
    conv1 = tf.nn.relu(conv1, 'relu')

    #conv1 = oneRun("conv3", conv1, convOutputSize=64,  kernelSize=3, stride=1)

    coarse6 = fc('coarse6', conv1, [4560, 4096], [4096], reuse=reuse, trainable=True)
    coarse7 = fc('coarse7', coarse6, [4096, 4070], [4070], reuse=reuse, trainable=True)
    coarse7_output = tf.reshape(coarse7, [8, 55, 74, 1])
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
