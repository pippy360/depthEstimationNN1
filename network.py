import tensorflow as tf

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
        #FIXME: Maybe replace the gpu here?
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
        #TODO: What are these biases for???? Why??? are they trained???
        bias = tf.nn.bias_add(conv, biases)
        conv_ = tf.nn.relu(bias, name=scope.name)
    return conv_

def inference():
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
