  # -*- coding: utf-8 -*-
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
import scipy
import numpy as np
import pandas as pd
import os

'''
SilCam TensorFlow analysis for classification of particle types
'''


def get_class_labels(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl'):
    '''
    Read the header file that defines the catagories of particles in the model

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle-classifier.tfl'

    Returns:
        class_labels (str)      : labelled catagories which can be predicted
     '''
    path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    class_labels = header.columns
    return class_labels


def load_model(model_path='/mnt/ARRAY/classifier/model/particle-classifier.tfl'):
    '''
    Load the trained tensorflow model

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle-classifier.tfl'

    Returns:
        model (tf model object) : loaded tfl model from load_model()
    '''
    path, filename = os.path.split(model_path)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    OUTPUTS = len(header.columns)
    class_labels = header.columns

    tf.reset_default_graph()

    # Same network definition as in tfl_tools scripts
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.75)
    network = fully_connected(network, OUTPUTS, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    model = tflearn.DNN(network, tensorboard_verbose=0,
            checkpoint_path=model_path)
    model.load(model_path)

    return model, class_labels

# Define the neural network
def build_model(IMXY='32', model_path='/mnt/ARRAY/classifier/model',
                model_file='plankton-classifier.tfl'):
    '''
    Build tensorflow model version 2

    Args:
        IMXY                    : input image size
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle-classifier.tfl'
        model_file              : model file name
    Returns:
        model (tf model object) : loaded tfl model from load_model()
        conv_arr                : convolution layer array for evaluation
        class_labels            : list of class labels
    '''
    check_point_file = os.path.join(model_path, model_file)
    print("check_point_file", check_point_file)
    header = pd.read_csv(os.path.join(model_path, 'header.tfl.txt'))
    OUTPUTS = len(header.columns)
    class_labels = header.columns

    print("Build the model...")
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    # Include the input layer, hidden layer(s), and set how you want to train the model
    inputsize = IMXY * IMXY * 3;
    # outputsize = pd.read_csv(header, header=None).shape[1] #np.shape(get_classes())[1]
    print("Inputlayer-size: %d" %(inputsize))

    # normalisation of images
    print("Normalisation of images...")
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping & rotating images
    print("Data augmentation...")
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    # Define the network architecture
    print("Define the network architecture...")
    net = input_data(shape=[None, IMXY, IMXY, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug);
    # 1: Convolution layer with 32 filters, each 3x3x3
    print('Step 1: Convolution layer with 32 filters, each 3x3x3')
    net = conv_2d(net, 32, 3, activation='relu', name='conv_1')
    conv_1 = net
    # 2: Max pooling layer
    print('Step 2: Max pooling')
    net = max_pool_2d(net, 2)
    # 3: Convolution layer with 64 filters
    print('Step 3: Convolution again')
    net = conv_2d(net, 64, 3, activation='relu')
    conv_2 = net


    # Step 4: Convolution yet again
    print('Step 3: Convolution yet again x4')
    net = conv_2d(net, 64, 3, activation='relu')
    conv_3 = net
    net = conv_2d(net, 64, 3, activation='relu')
    conv_4 = net
    net = conv_2d(net, 64, 3, activation='relu')
    conv_5 = net
    net = conv_2d(net, 64, 3, activation='relu')
    conv_6 = net

    # Step 5: Max pooling layer
    print('Step 5: Max pooling')
    net = max_pool_2d(net, 2)

    # Step 6: Fully-connected 512 node neural network
    print('Step 6: Fully-connected 512 node neural network')
    net = fully_connected(net, 512, activation='relu')

    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    print('Step 7: Dropout - throw away some data randomly during training to prevent over-fitting')
    net = dropout(net, 0.75)

    # Step 8: Fully-connected neural network with outputs to make the final prediction
    net = fully_connected(net, (OUTPUTS+1), activation='softmax')
    #net = fully_connected(net, (OUTPUTS), activation='softmax')

    # Tell tflearn how we want to train the network
    net = regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    # Wrap the network in a model object
    model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=check_point_file)

    #if mode == 'evaluate':
        #model.load(check_point_file)
    conv_arr = [conv_1, conv_2, conv_3, conv_4, conv_5, conv_6]
    return model, conv_arr, class_labels

def predict(img, model):
    '''
    Use tensorflow model to classify particles
    
    Args:
        img (uint8)             : a particle ROI, corrected and treated with the silcam
                                  explode_contrast function
        model (tf model object) : loaded tfl model from load_model()

    Returns:
        prediction (array)      : the probability of the roi belonging to each class
    '''

    # Scale it to 32x32
    img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

    # Predict
    prediction = model.predict([img])

    return prediction


def build_alexnet(IMXY='32', model_path='/mnt/ARRAY/classifier/model',
                model_file='plankton-classifier.tfl'):
    '''
    build AlexNet model

    Args:
        IMXY                    : input image size
        model_path (str)        : path to particle-classifier e.g.
                                  '/mnt/ARRAY/classifier/model/particle-classifier.tfl'
        model_file              : model file name
    Returns:
        model (tf model object) : loaded tfl model from load_model()
        conv_arr                : convolution layer array for evaluation
        class_labels            : list of class labels
    '''
    check_point_file = os.path.join(model_path, model_file)
    print("check_point_file", check_point_file)
    header = pd.read_csv(os.path.join(model_path, 'header.tfl.txt'))
    OUTPUTS = len(header.columns)
    class_labels = header.columns

    print("Build the model...")
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    # Include the input layer, hidden layer(s), and set how you want to train the model
    inputsize = IMXY * IMXY * 3;
    # outputsize = pd.read_csv(header, header=None).shape[1] #np.shape(get_classes())[1]
    print("Inputlayer-size: %d" %(inputsize))

    # normalisation of images
    print("Normalisation of images...")
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping & rotating images
    print("Data augmentation...")
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    # Define the network architecture
    print("Define the network architecture...")
    net = input_data(shape=[None, IMXY, IMXY, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug);
    # 1: Convolution layer with 32 filters, each 3x3x3
    print('Step 1: Convolution layer with 32 filters, each 3x3x3')
    net = conv_2d(net, 32, 11, strides=4, activation='relu')
    #net = conv_2d(net, 32, 3, activation='relu', name='conv_1')
    conv_1 = net
    # 2: Max pooling layer
    print('Step 2: Max pooling')
    net = max_pool_2d(net, 2)
    # 3: Convolution layer with 64 filters
    print('Step 3: Convolution again')
    net = conv_2d(net, 64, 3, activation='relu')
    conv_2 = net


    # Step 4: Convolution yet again
    print('Step 3: Convolution yet again x4')
    net = conv_2d(net, 64, 3, activation='relu')
    conv_3 = net
    net = conv_2d(net, 64, 3, activation='relu')
    conv_4 = net
    net = conv_2d(net, 64, 3, activation='relu')
    conv_5 = net
    net = conv_2d(net, 64, 3, activation='relu')
    conv_6 = net

    # Step 5: Max pooling layer
    print('Step 5: Max pooling')
    net = max_pool_2d(net, 2)

    # Step 6: Fully-connected 512 node neural network
    print('Step 6: Fully-connected 512 node neural network')
    net = fully_connected(net, 512, activation='relu')

    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    print('Step 7: Dropout - throw away some data randomly during training to prevent over-fitting')
    net = dropout(net, 0.75)

    # Step 8: Fully-connected neural network with outputs to make the final prediction
    net = fully_connected(net, (OUTPUTS+1), activation='softmax')
    #net = fully_connected(net, (OUTPUTS), activation='softmax')

    # Tell tflearn how we want to train the network
    net = regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    # Wrap the network in a model object
    model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=check_point_file)

    #if mode == 'evaluate':
        #model.load(check_point_file)
    conv_arr = [conv_1, conv_2, conv_3, conv_4, conv_5, conv_6]
    return model, conv_arr, class_labels