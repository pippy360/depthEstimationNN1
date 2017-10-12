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
            'weights1',
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

def oneRun(scope_name, inputs, convOutputSize, inChannels, kernelSize=3, stride=1):
    with tf.variable_scope(scope_name) as scope:
        wd = 0.0
        conv1 = conv2d('conv1', inputs, [kernelSize, kernelSize, inChannels, convOutputSize], [convOutputSize], [1, stride, stride, 1], padding='SAME', trainable=True)
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

def oneRunWithoutRelu(scope_name, inputs, convOutputSize, inChannels, kernelSize=3, stride=1):
    with tf.variable_scope(scope_name) as scope:
        wd = 0.0
        conv1 = conv2d('conv1', inputs, [kernelSize, kernelSize, inChannels, convOutputSize], [convOutputSize], [1, stride, stride, 1], padding='SAME', trainable=True)
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

def resizeLayer(scope_name, inputs, initInputSize, smallSize, bigSize, stride=1):
    
    with tf.variable_scope(scope_name) as scope:
        conv1 = oneRun("conv2", inputs, convOutputSize=smallSize, inChannels=initInputSize, kernelSize=1, stride=stride)
        conv1 = oneRun("conv3", conv1, convOutputSize=smallSize, inChannels=smallSize,  kernelSize=3, stride=1)
        conv1 = oneRunWithoutRelu("conv4", conv1, convOutputSize=bigSize, inChannels=smallSize, kernelSize=1, stride=1)

        #resize the original input
        conv1b = oneRunWithoutRelu("conv5", inputs, convOutputSize=bigSize, inChannels=initInputSize, kernelSize=1, stride=stride)

        #concat
        conv1 = conv1 + conv1b
        conv1 = tf.nn.relu(conv1, 'relu')

        return conv1

def nonResizeLayer(scope_name, inputs, initInputSize, smallSize, bigSize):

    with tf.variable_scope(scope_name) as scope:
        conv1 = oneRun("conv2", inputs, convOutputSize=smallSize, inChannels=initInputSize, kernelSize=1, stride=1)
        conv1 = oneRun("conv3", conv1, convOutputSize=smallSize, inChannels=smallSize,  kernelSize=3, stride=1)
        conv1 = oneRunWithoutRelu("conv4", conv1, convOutputSize=bigSize, inChannels=smallSize, kernelSize=1, stride=1)

        #concat
        conv1 = conv1 + inputs
        conv1 = tf.nn.relu(conv1, 'relu')

        return conv1

def inference(images, reuse=False, trainable=True):

    #first bit
    conv1 = oneRun("conv1", images, convOutputSize=64, inChannels=3, kernelSize=7, stride=2)
    conv1b = maxPool("max1", conv1, kernelSize=3, stride=2)

    conv1 = resizeLayer("resize1", conv1b, initInputSize=64, smallSize=64, bigSize=256)
    print "conv1"
    print conv1
    
    for i in range(2):
        conv1 = nonResizeLayer("resize2"+str(i), conv1, initInputSize=256, smallSize=64, bigSize=256)
    
    conv1 = resizeLayer("resize3", conv1, initInputSize=256, smallSize=128, bigSize=512, stride=2)

    l1concat = conv1
    print "l1concat"
    print l1concat

    for i in range(7):
        conv1 = nonResizeLayer("resize4"+str(i), conv1, initInputSize=512, smallSize=128, bigSize=512)
    
    l2concat = conv1
    print "l2concat"
    print l2concat

    conv1 = resizeLayer("resize5", conv1, initInputSize=512, smallSize=256, bigSize=1024)

    l3concat = conv1
    print "l3concat"
    print l3concat

    for i in range(35):
        conv1 = nonResizeLayer("resize6"+str(i), conv1, initInputSize=1024, smallSize=256, bigSize=1024)

    l4concat = conv1
    print "l4concat"
    print l4concat

    conv1 = resizeLayer("resize7", conv1, initInputSize=1024, smallSize=512, bigSize=2048)

    l5concat = conv1
    print "l5concat"
    print l5concat

    for i in range(2):
        conv1 = nonResizeLayer("resize8"+str(i), conv1, initInputSize=2048, smallSize=512, bigSize=2048)

    l6concat = conv1
    print "l6concat"
    print l6concat

    conv1 = tf.concat([l1concat, l2concat, l3concat, l4concat, l5concat, l6concat], 3)

    conv1 = tf.layers.dropout(conv1, .5)

    conv1 = conv2d('convFinal', conv1, [3, 3, 7168, 200], [200], [1, 1, 1, 1], padding='SAME', trainable=True)
        
    conv1 = tf.layers.conv2d_transpose(conv1, 1, 8, strides=(4, 4), padding='SAME')

    return conv1

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
