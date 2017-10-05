import tensorflow as tf

def conv2d(scope_name, inputs, shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True):
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
    return conv_

def inference():

    #BLOCK 0
    block1Output = ''#FIXME:::::::::
    {
        blockToken1 = conv2d("Block0", images, shape=[7, 7, 3, 64], stride=[1, 2, 2, 1], padding='SAME')
        blockToken2 = tf.contrib.layers.batch_norm(blockToken1)
        scaleFactor = _variable_with_weight_decay(
            'scaleWeight',
            shape=[1],
            stddev=0.01,
            wd=0.0,
            trainable=trainable
        )
        blockToken3 = tf.multiply(blockToken2, scaleFactor)
        addFactor = _variable_with_weight_decay(
            'addWeight',
            shape=[1],
            stddev=0.01,
            wd=0.0,
            trainable=trainable
        )
        blockToken4 = tf.add(blockToken3, addFactor)
        blockToken5 = tf.nn.relu(blockToken4, name=scope.name)
        block1Output = blockToken5
    }

    return block1Output

print inference()
