import tensorflow as tf

#Defining Gaussian Kernel(4-d) Variable
def weight(shape):
    init=tf.truncated_normal(shape,stddev=1e-4)
    return tf.Variable(init)

#Defining Bias Variable
def bias(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)

#Convolution(2-d)
def conv2d(input,kernel,padding):
    return tf.nn.conv2d(input,kernel,strides=[1,1,1,1],padding=padding)

#Max-Pooling(2-d)
def max_pool_2x2(input):
    return tf.nn.max_pool(input,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")


# local layer weight initialization
def local_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)

def local_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)