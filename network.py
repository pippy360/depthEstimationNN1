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
    {
        blockToken1 = conv2d("Block0", images, [7, 7, 3, 64], [1, 2, 2, 1], padding='SAME')
        blockToken2 = tf.nn.relu(blockToken1, name=scope.name)
        block0MaxPool = tf.nn.max_pool(blockToken2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    }

    #BLOCK 1
    {
        risidualOriginal
        convToken2 = conv2d("Block1", images, [1, 1, 3, 64], [1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        convToken3 = conv2d("Block1", images, [7, 7, 3, 64], [1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        convToken3 = conv2d("Block1", images, [7, 7, 3, 64], [1, 2, 2, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(convToken3, risidualOriginal)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    }
    
    #BLOCK 2
    {
    
    }
    
    #convolution from the paper might include a relu layer
    
    #input in an image 240x320x3
    
    conv(kernal=[1,7,7,1], outDepth=64, stride=2)#are we sure 64 is output depth????
    maxPool([3,3], outDepth=whatGoesHere)  
    
    #block1
    #input is 240/2 ????
    #output
    
    #final output of the image is 120x160 (so the stride is only /2 once)
    
    return ""

print inference()
