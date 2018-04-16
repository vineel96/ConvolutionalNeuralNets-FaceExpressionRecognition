import tensorflow as tf
import CNN_Layers as layer
import numpy as np

labels_count=7 # total 7 expressions
LEARNING_RATE=1e-4
TRAINING_ITERATIONS=3000
DROPOUT=0.5
BATCH_SIZE=50

VALIDATION_SIZE = 1709
validation_images=""
validation_labels=""

def train_CNN_Model():


    '''Defining CNN Model using Tensorflow'''

    # Input And Output of NN
    input = tf.placeholder('float', shape=[None, 2304],name="Input")  # Images
    output = tf.placeholder('float', shape=[None, labels_count])  # Labels

    # First Convolutional Layer 64

    kernel_conv1 = layer.weight([5, 5, 1, 64])  # kernels are nothing but weights in neural nets
    bias_conv1 = layer.bias([64])
    image = tf.reshape(input, [-1, 48, 48, 1])
    conv1 = tf.nn.relu(layer.conv2d(image, kernel_conv1, "SAME") + bias_conv1)
    pool1 = layer.max_pool_2x2(conv1)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # Second Convolutional Layer
    kernel_conv2 = layer.weight([5, 5, 64, 128])
    bias_conv2 = layer.bias([128])
    conv2 = tf.nn.relu(layer.conv2d(norm1, kernel_conv2, "SAME") + bias_conv2)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = layer.max_pool_2x2(norm2)

    # densely connected layer-1 local 3
    weight_fc1 = layer.local_weight_variable([12 * 12  * 128, 3072])
    bias_fc1 = layer.local_bias_variable([3072])
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 128])  # reshape to 2-d vector (27000,12,12,128)=>(27000,12*12*128)
    fc1 = tf.nn.relu(tf.matmul(pool2_flat, weight_fc1) + bias_fc1)  # (27000,3072)

    # densely connected layer-2 local 4

    weight_fc2 = layer.local_weight_variable([3072, 1536])
    bias_fc2 = layer.local_bias_variable([1536])
    fc1_flat = tf.reshape(fc1, [-1, 3072])
    fc2 = tf.nn.relu(tf.matmul(fc1_flat, weight_fc2) + bias_fc2)  # (-1,1536)

    # dropout
    keep_prob = tf.placeholder('float')
    fc2_drop = tf.nn.dropout(fc2, keep_prob)

    # Final Layer for Deep Net
    weight_fc3 = layer.weight([1536, labels_count])
    bias_fc3 = layer.bias([labels_count])
    y = tf.nn.softmax(tf.matmul(fc2_drop, weight_fc3) + bias_fc3,name="final_result")  # (-1,7)

    # cost function
    cross_entropy = -tf.reduce_sum(output * tf.log(y))

    # optimization function
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    predict = tf.argmax(y, 1)

    predict_all=y

    '''Running Defined Model Using TensorFlow Session'''

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    for i in range(TRAINING_ITERATIONS):

        # get new batch
        batch_xs, batch_ys = next_batch(BATCH_SIZE)
        # Evaluation our model using train data and validation data
        train_accuracy = accuracy.eval(feed_dict={input: batch_xs,
                                                      output: batch_ys,
                                                      keep_prob: 1.0})

        if (VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={input: validation_images[0:BATCH_SIZE],
                                                               output: validation_labels[0:BATCH_SIZE],
                                                               keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
                    train_accuracy, validation_accuracy, i))
        else:
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))

        # Train our model
        sess.run(train_step, feed_dict={input: batch_xs, output: batch_ys, keep_prob: DROPOUT})


def next_batch(batch_size):
    global train_images,train_labels,index_in_epoch,epochs_completed,num_examples
    start=index_in_epoch
    index_in_epoch+=batch_size

    #when all training data have been already used,it is reorder randomly
    if index_in_epoch>num_examples:
        epochs_completed+=1 #finished epoch
        #shuffle the data
        perm=np.arange(num_examples)
        np.random.shuffle(perm)
        train_images=train_images[perm]
        train_labels=train_labels[perm]
        #start next epoch
        start=0
        index_in_epoch=batch_size
        assert batch_size<=num_examples
    end =index_in_epoch
    return train_images[start:end],train_labels[start:end]