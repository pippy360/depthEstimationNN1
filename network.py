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

    #BLOCK 0 ... there must always be a RELU between convolutions
    convToken1 = conv2d("Block0", images, [7, 7, 3, 64], [1, 2, 2, 1], padding='SAME')
    convToken2 = tf.nn.max_pool(convToken1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    #BLOCK 1
    convToken2 = conv2d("Block1", images, [1, 1, 3, 64], [1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    convToken3 = conv2d("Block1", images, [7, 7, 3, 64], [1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    convToken3 = conv2d("Block1", images, [7, 7, 3, 64], [1, 2, 2, 1], padding='SAME')
    #some sort 
    #add them now before the activation function
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
    
    convToken2 = conv2d("Block0", images, [5, 5, 3, 64], [1, 2, 2, 1], padding='SAME')
    convToken3 = conv2d("Block0", images, [5, 5, 3, 64], [1, 2, 2, 1], padding='SAME')

    #convolution from the paper might include a relu layer
    
    #input in an image 240x320x3
    
    conv(kernal=[1,7,7,1], outDepth=64, stride=2)#are we sure 64 is output depth????
    maxPool([3,3], outDepth=whatGoesHere)  
    
    #block1
    #input is 240/2 ????
    #output
    
    #how do we do the skipping?
    conv([1,1], outDepth=64)
    conv([3,3], outDepth=64)
    conv([1,1], outDepth=256)
    
    conv([1,1], outDepth=64)
    conv([3,3], outDepth=64)
    conv([1,1], outDepth=256)
    
    #block repeated twice, which same size input and output
    
    #final output of the image is 120x160 (so the stride is only /2 once)
    #/2 == 2 stride, /8 == 8 stride
    
    return ""

print inference()
