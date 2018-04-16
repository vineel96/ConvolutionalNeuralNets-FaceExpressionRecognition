import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import numpy as np
import pandas as pd

def read_train_data():
    global pixels, data, train_data,pixels1,train1
    data = pd.read_csv("fer2013.csv")


    train_data = data[data.Usage == "Training"]
    train1=data[data.Usage == "PrivateTest"]
    pixels1=train1.pixels.str.split(" ")
    pixels = train_data.pixels.str.split(" ")



# Preprocessing the image data-sets
def preprocees():
    global pixels, train_data, images_data, labels_count, train_images, train_labels, validation_images, validation_labels,test_images,test_labels
    global labels_flat,labels_test_flat,pixels1,train1
    pixels = pixels.tolist()
    images_data = np.array(pixels)
    images_data = images_data.astype(np.float)

    pixels1 = pixels1.tolist()
    im=np.array(pixels1)
    im=im.astype(np.float)

    labels_flat = train_data["emotion"].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    num_labels = labels_flat.shape[0]
    index_offset = np.arange(num_labels) * labels_count
    labels_one_hot = np.zeros((num_labels, labels_count))
    labels_one_hot.flat[index_offset + labels_flat.ravel()] = 1
    labels_one_hot.astype(np.uint8)

    labels_flat1 = train1["emotion"].values.ravel()
    labels_flat=np.concatenate((labels_flat,labels_flat1),axis=0)

    num_labels1 = labels_flat1.shape[0]
    index_offset1 = np.arange(num_labels1) * labels_count
    labels_one_hot1 = np.zeros((num_labels1, labels_count))
    labels_one_hot1.flat[index_offset1 + labels_flat1.ravel()] = 1
    labels_one_hot1.astype(np.uint8)

    train_images = np.concatenate((images_data,im),axis=0)
    train_labels = np.concatenate((labels_one_hot,labels_one_hot1),axis=0)



    print("\nNumber of Unique facial expressions is %d\n" % labels_count)
    print("\nno of img examples %d\n " % train_images.shape[0])


    test_data = data[data.Usage == "PublicTest"]
    labels_test_flat=test_data["emotion"].values.ravel()
    labels_count = np.unique(labels_test_flat).shape[0]
    num_labels_test = labels_test_flat.shape[0]
    index_offset_test = np.arange(num_labels_test) * labels_count
    labels_one_hot_test = np.zeros((num_labels_test, labels_count))
    labels_one_hot_test.flat[index_offset_test + labels_test_flat.ravel()] = 1
    labels_one_hot_test.astype(np.uint8)


    print(len(test_data))
    test_pixels_values = test_data.pixels.str.split(" ").tolist()
    test_pixels_values = pd.DataFrame(test_pixels_values, dtype=int)
    test_images = test_pixels_values.values
    test_images = test_images.astype(np.float)  # only use for single test array.astype


    test_labels=labels_one_hot_test


if __name__ == "__main__":
    n = 5
    global train_images ,train_labels ,labels_flat ,test_images ,test_labels ,labels_test_flat


    #This function is used to read the data from .CSV file
    read_train_data()

    #This function is used to do some basic preprocessing steps
    preprocees()


    X = train_images
    Y = labels_flat


    X_test = test_images
    Y_test = labels_test_flat



    X = X.reshape([-1, 48, 48, 1])
    X_test = X_test.reshape([-1, 48, 48, 1])


    Y = tflearn.data_utils.to_categorical(Y, 7)
    Y_test = tflearn.data_utils.to_categorical(Y_test, 7)


    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()


    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()



    net = tflearn.input_data(shape=[None ,48 ,48 ,1] ,data_preprocessing=img_prep, data_augmentation=img_aug
                             ,name='Input')
    net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, n- 1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, n - 1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)

    net = tflearn.fully_connected(net, 7, activation='softmax', name='final_result')
    mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.0001, decay_step=32000, staircase=True, momentum=0.9)
    net = tflearn.regression(net, optimizer=mom,
                             loss='categorical_crossentropy', name='final_result1')

    # Training
    model = tflearn.DNN(net, checkpoint_path='./new_trained_models/project_directory/model.tflearn', max_checkpoints=50,
                        tensorboard_verbose=0, clip_gradients=0.)


    model.fit(X, Y, n_epoch=150, snapshot_epoch=False, snapshot_step=1000, show_metric=True, batch_size=100, shuffle=True,run_id='resnet_emotion_sample')

    score = model.evaluate(X_test, Y_test)
    print("\n")
    print('Test Accuarcy: ', score)
    model.save('./model.tflearn')


