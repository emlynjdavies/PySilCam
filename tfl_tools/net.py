# -*- coding: utf-8 -*-

#
from sklearn import metrics

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
import numpy as np

'''
Deep Neural Network Architecture
build - train - predict - evaluate 
'''

class Net:

    def __init__(self, name='LeNet', input_width=32, input_height=32, input_channels=3, num_classes=7, learning_rate=0.01,
                 momentum=0.9, keep_prob=0.8,
                 model_file='plankton-classifier.tfl'):
        '''
        Network Initialization
        :param name:            Name of the model (LeNet, CIFAR10, AlexNet, VGGNet, ResNet)
        :param input_width:     input image width default (32)
        :param input_height:    input image width default (32)
        :param input_channels:  input image number of channels default (3)
        :param num_classes:     number of classes
        :param learning_rate:   Learing rate (default 0.01)
        :param momentum:        Momentum (not used yet... consider use/remove)
        :param keep_prob:       drop out probability (default 0.8)
        :param model_file:      the name of the saved model file - it should be passed as absolute link
        '''
        self.name = name

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.num_classes = num_classes

        self.learning_rate = learning_rate

        self.momentum = momentum
        self.keep_prob = keep_prob

        self.random_mean = 0
        self.random_stddev = 0.01
        self.check_point_file = model_file

    def __preprocessing(self):
        '''
        Image preprocessing
        :return: img_prep - preprocessed images
        '''
        # normalisation of images
        print("Normalisation of images...")
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()
        return img_prep

    def __data_augmentation(self):
        '''
        Data Augmentation
        flip left and right
        rotation
        blur
        :return: img_aug - image augmentation
        '''
        # Create extra synthetic training data by flipping & rotating images
        print("Data augmentation...")
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)
        img_aug.add_random_blur(sigma_max=3.)
        return img_aug

    def train(self,model, trainX, trainY, testX, testY,
              round_num='01', n_epoch=50, batch_size=128,
              model_name = 'plankton-classifier'):
        '''
        Training the model
        :param model:       The model to be trained
        :param trainX:      Input training set
        :param trainY:      Ouput true training set
        :param testX:       Input test set
        :param testY:       Output test set
        :param round_num:   round number in case the cross validation
        :param n_epoch:     Number of epochs
        :param batch_size:  batch size
        :return:
        '''
        model.fit(trainX, trainY, n_epoch=n_epoch, shuffle=True, validation_set=(testX, testY),
                  show_metric=True, batch_size=batch_size,
                  snapshot_epoch=True,
                  run_id=model_name + round_num)



    def evaluate(self, model, testX, testY):
        '''
        Evaluate the model
        :param model:   The model to be evaluated
        :param testX:   Input test set
        :param testY:   Output test set
        :return: summaries for y_pred, y_true,
                    accuracy, precision, recall, f1_score,
                    confusion_matrix, normalized_confusion_matrix
        '''

        # print("\nTest prediction for x = ", testX)
        print("model evaluation ")
        print('testX.type',type(testX))
        print('testY.type', type(testY))
        #predictions = model.predict(testX)
        # predictions = [int(i) for i in model.predict(testX)]
        #print("predictions: ", predictions)
        '''

        y_pred = []
        #for pred in predictions:
        for x in testX:
            pred = model.predict([x])
            y_pred.append(pred.argmax(axis=0))
            print("prediction x, np.argmax(pred): ", pred, pred.argmax(axis=0))
        print(y_pred)
        print("testY: ")
        y_true = []
        for ty in testY:
            y_true.append(ty.argmax(axis=0))
            print("ty, y_true: ", ty, ty.argmax(axis=0))
        print(y_true)
        '''
        y_pred = []
        y_true = []
        pred = model.predict(testX)
        for ty in pred:
            print("ty, y_pred: ", ty, ty.argmax(axis=0))
            y_pred.append(ty.argmax(axis=0))
        for ty in testY:
            print("ty, y_true: ", ty, ty.argmax(axis=0))
            y_true.append(ty.argmax(axis=0))

        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("Accuracy: {}%".format(100 * accuracy))

        precision = metrics.precision_score(y_true, y_pred, average="weighted")
        print("Precision: {}%".format(100 * precision))

        recall = metrics.recall_score(y_true, y_pred, average="weighted")
        print("Recall: {}%".format(100 * recall))

        f1_score = metrics.f1_score(y_true, y_pred, average="weighted")
        print("f1_score: {}%".format(100 * f1_score))

        print("Confusion Matrix:")
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        print(confusion_matrix)
        normalized_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
        print("")
        print("Confusion matrix (normalised to % of total test data):")
        print(normalized_confusion_matrix)
        return y_pred, y_true, accuracy, precision, recall, f1_score, confusion_matrix, normalized_confusion_matrix


    def build_model(self, model_file):
        '''
        Build the model depends on the initialize network name
        LetNet - based on LeCun
        MINSTNet - based on LeCun network used to classify the MINST dataset
        CIFAR10 - the version of the LeNet used to classify the CIFAR dataset for 10 classes
        AlexNet - the version of AlexNet reconstructed to be compatible with the input images
        VGGNet -
        ResNet -
        :param model_file: the model file
        :return: The model and and convolution array
        '''
        print(self.name)
        self.model_file = model_file
        print(self.model_file)
        if self.name == 'OrgNet':
            return self.__build_OrgNet()
        if self.name == 'LeNet':
            return self.__build_LeNet()
        elif self.name == 'MINST':
            return self.__build_MINST()
        elif self.name == 'CIFAR10':
            return self.__build_CIFAR10()
        elif self.name == 'AlexNet':
            return self.__build_AlexNet()
        elif self.name == 'VGGNet':
            return self.__build_VGGNet()
        elif self.name == 'GoogLeNet':
            return self.__build_GoogLeNet()
        elif self.name == 'ResNet':
            return self.__build_ResNet()
        elif self.name == 'ResNeXt':
            return self.__build_ResNeXt()
        elif self.name == 'PlankNet':
            return self.__build_PlankNet()
        elif self.name == 'CoapNet':
            print(self.name)
            return self.__build_CoapNet()

    def __build_OrgNet(self):
        '''
        Build the model based on LeCun proposed architecture
        :return: The model and and convolution array
        '''
        print("Building" + self.name + " model ...")
        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        # Layer 1
        print('Layer 1: Convolution layer with 32 filters, each 3x3x3')
        # 1: Convolution layer with 32 filters, each 3x3x3
        # incoming, number of filters, filter size, strides, padding, activation, bias, weigths_init, bias_init,
        # regularizer, weight_decay
        net = conv_2d (net, 32, 3, activation='relu', name='conv_1')
        conv_1 = net
        # 2: Max pooling layer
        print('  2: Max pooling')
        net = max_pool_2d(net, 2)

        # Layer 2:
        print('Layer 2:')
        # 3: Convolution layer with 16 filters size 5 and stride 1
        print('1: Convolution again')
        net = conv_2d(net, 64, 3, activation='relu', name='conv_2')
        conv_2 = net

        # Step 4: Convolution yet again
        print('Step 3: Convolution yet again x4')
        net = conv_2d(net, 64, 3, activation='relu', name='conv_3')
        conv_3 = net
        net = conv_2d(net, 64, 3, activation='relu', name='conv_4')
        conv_4 = net
        net = conv_2d(net, 64, 3, activation='relu', name='conv_5')
        conv_5 = net
        net = conv_2d(net, 64, 3, activation='relu', name='conv_6')
        conv_6 = net

        # Step 5: Max pooling layer
        print('Step 5: Max pooling')
        net = max_pool_2d(net, 2)

        # Step 6: Fully-connected 512 node neural network
        print('Step 6: Fully-connected 512 node neural network')
        net = fully_connected(net, 512, activation='relu')

        # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
        #print('Step 7: Dropout - throw away some data randomly during training to prevent over-fitting')
        #net = dropout(net, self.keep_prob)

        # Step 8: Fully-connected neural network with outputs to make the final prediction
        net = fully_connected(net, self.num_classes+1, activation='softmax')

        net = regression(net, optimizer='adam', learning_rate=self.learning_rate,
                             loss='categorical_crossentropy', name='target')

        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2, conv_3, conv_4, conv_5, conv_6]
        return model, conv_arr

    def __build_CoapNet(self):
        '''
        Build the model based on CoapNet proposed architecture
        :return: The model and and convolution array
        '''
        print("Building" + self.name + " model ...")
        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        # Layer 1
        print('Layer 1: Convolution layer with 32 filters, each 3x3x3')
        # 1: Convolution layer with 64 filters, each 3x3x3
        # incoming, number of filters, filter size, strides, padding, activation, bias, weigths_init, bias_init,
        # regularizer, weight_decay
        net = conv_2d(net, 64, 3, activation='relu', name='conv_1')
        conv_1 = net
        # 2: Max pooling layer
        print('  2: Max pooling')
        net = max_pool_2d(net, 2)

        # Layer 2:
        print('Layer 2:')
        # 3: Convolution layer with 128 filters size 3 and stride 1
        print('1: Convolution again')
        net = conv_2d(net, 128, 3, activation='relu', name='conv_2')
        conv_2 = net
        net = max_pool_2d(net, 2)

        # Layer 3
        print('Layer 3: Convolution layer with 256 filters, each 3x3x3')
        # 3: Convolution layer with 256 filters, each 3x3x3
        net = conv_2d(net, 256, 3, activation='relu', name='conv_3')
        conv_3 = net
        net = max_pool_2d(net, 2)

        # Layer 4
        print('Layer 4: Convolution layer with 512 filters, each 3x3x3')
        # 4: Convolution layer with 512 filters, each 3x3x3
        net = conv_2d(net, 512, 3, activation='relu', name='conv_4')
        conv_4 = net
        net = max_pool_2d(net, 2)

        # Step 6: Fully-connected 512 node neural network
        print('Step 6: Fully-connected 512 node neural network')
        net = fully_connected(net, 512, activation='relu')
        # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
        #print('Step 7: Dropout - throw away some data randomly during training to prevent over-fitting')
        #net = dropout(net, self.keep_prob)
        net = fully_connected(net, 256, activation='relu')
        # Step 8: Dropout - throw away some data randomly during training to prevent over-fitting
        #print('Step 8: Dropout - throw away some data randomly during training to prevent over-fitting')
        #net = dropout(net, self.keep_prob)
        net = fully_connected(net, 256, activation='relu')
        # Step 9: Dropout - throw away some data randomly during training to prevent over-fitting
        print('Step 9: Dropout - throw away some data randomly during training to prevent over-fitting')
        #net = dropout(net, self.keep_prob)

        # Step 8: Fully-connected neural network with outputs to make the final prediction
        net = fully_connected(net, self.num_classes + 1, activation='softmax')

        net = regression(net, optimizer='adam', learning_rate=self.learning_rate,
                         loss='categorical_crossentropy', name='target')

        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2, conv_3, conv_4]
        return model, conv_arr

    def __build_LeNet(self):
        '''
        Build the model based on LeCun proposed architecture
        :return: The model and and convolution array
        '''
        print("Building" + self.name + " model ...")
        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        # Layer 1
        print('Layer 1: Convolution layer with 32 filters, each 3x3x3')
        # 1: Convolution layer with 32 filters, each 3x3x3
        print('  1: Convolution layer with 6 filters, each 5x5x3')
        # incoming, number of filters, filter size, strides, padding, activation, bias, weigths_init, bias_init,
        # regularizer, weight_decay
        net = conv_2d (net, 6, 5, activation='relu', regularizer="L2", name='conv_1')
        conv_1 = net
        # 2: Max pooling layer
        print('  2: Max pooling')
        # kernel size, stride, padding

        net = max_pool_2d(net, 2, 2)

        # Layer 2:
        print('Layer 2:')
        # 3: Convolution layer with 16 filters size 5 and stride 1
        print('1: Convolution again')
        net = conv_2d(net, 16, 5, activation='relu', regularizer="L2", name='conv_2')
        conv_2 = net
        # 2: Max pooling layer
        print('  2: Max pooling')
        net = max_pool_2d(net, 2, 2)

        # Layer 3: Fully-connected 128 node neural network
        print('Layer 3: Fully-connected 120 node neural network')
        net = fully_connected(net, 120, activation='tanh')
        net = dropout(net, self.keep_prob)

        # Layer 4: Fully-connected 256 node neural network
        print('Layer 4: Fully-connected 84 node neural network')
        net = fully_connected(net, 84, activation='tanh')

        # Layer 5: Fully-connected 256 node neural network
        print('Layer 5: Fully-connected number of classes node neural network')
        net = fully_connected(net, self.num_classes+1, activation='softmax')

        net = regression(net, optimizer='adam', learning_rate=self.learning_rate,
                             loss='categorical_crossentropy', name='target')

        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2]
        return model, conv_arr

    def __build_MINST(self):
        '''
        Build the model
        MINSTNet - based on LeCun network used to classify the MINST dataset
        :return: The model and and convolution array
        '''
        print("Building" + self.name + " model ...")
        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        # Layer 1
        print('Layer 1: Convolution layer with 32 filters, each 3x3x3')
        # 1: Convolution layer with 32 filters, each 3x3x3
        print('  1: Convolution layer with 32 filters, each 3x3x3')
        net = conv_2d(net, 32, 3, activation='relu', regularizer="L2", name='conv_1')
        conv_1 = net
        # 2: Max pooling layer
        print('  2: Max pooling')
        net = max_pool_2d(net, 2)
        # 3: local_response_normalization
        print('  3: Local Response Normalization')
        net = local_response_normalization(net)
        # Layer 2:
        print('Layer 2:')
        # 3: Convolution layer with 64 filters
        print('1: Convolution again')
        net = conv_2d(net, 64, 3, activation='relu', regularizer="L2", name='conv_2')
        conv_2 = net
        # 2: Max pooling layer
        print('  2: Max pooling')
        net = max_pool_2d(net, 2)
        # 3: local_response_normalization
        print('  3: Local Response Normalization')
        net = local_response_normalization(net)

        # Layer 3: Fully-connected 128 node neural network
        print('Layer 3: Fully-connected 128 node neural network')
        net = fully_connected(net, 128, activation='tanh')
        net = dropout(net, self.keep_prob)

        # Layer 4: Fully-connected 256 node neural network
        print('Layer 4: Fully-connected 256 node neural network')
        net = fully_connected(net, 256, activation='tanh')
        net = dropout(net, self.keep_prob)

        # Layer 5: Fully-connected 256 node neural network
        print('Layer 5: Fully-connected number of classes node neural network')
        net = fully_connected(net, self.num_classes+1, activation='softmax')

        net = regression(net, optimizer='adam', learning_rate=self.learning_rate,
                             loss='categorical_crossentropy', name='target')

        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2]
        return model, conv_arr

    def __build_CIFAR10(self):
        '''
        Build the model
        CIFAR10 - the version of the LeNet used to classify the CIFAR dataset for 10 classes
        :return: The model and and convolution array
        '''
        print("Building" + self.name + " model ...")

        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        # Layer 1
        print('Layer 1: Convolution layer with 32 filters, each 3x3x3')
        # 1: Convolution layer with 32 filters, each 3x3x3
        print('  1: Convolution layer with 32 filters, each 3x3x3')
        net = conv_2d(net, 32, 3, activation='relu', name='conv_1')
        conv_1 = net
        # 2: Max pooling layer
        print('  2: Max pooling')
        net = max_pool_2d(net, 2)
        # Layer 2:
        print('Layer 2:')
        # 3: Convolution layer with 64 filters
        print('1: Convolution again')
        net = conv_2d(net, 64, 3, activation='relu', name='conv_2')
        conv_2 = net
        print('Layer 3:')
        # 3: Convolution layer with 64 filters
        print('1: yet another Convolution ')
        net = conv_2d(net, 64, 3, activation='relu', name='conv_3')
        conv_3 = net
        # 2: Max pooling layer
        print('  2: Max pooling')
        net = max_pool_2d(net, 2)

        # Layer 4: Fully-connected 512 node neural network
        print('Layer 4: Fully-connected 512 node neural network')
        net = fully_connected(net, 512, activation='relu')
        net = dropout(net, self.keep_prob)  # keep_prob = 0.5

        # Layer 5: Fully-connected 10 number of classes
        print('Layer 5: Fully-connected number of classes node neural network')
        net = fully_connected(net, self.num_classes + 1, activation='softmax')

        net = regression(net, optimizer='adam', learning_rate=self.learning_rate,  # learning_rate=0.001
                         loss='categorical_crossentropy', name='target')

        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2, conv_3]
        return model, conv_arr

    def __build_PlankNet(self):
        '''
        Build the model
        PlanktonNet - ZooplanktonNet: (Dai et al. 2016) achieve 93.7% accuracy
        Architecture:
            - For the 1st 3 conv layer bet 13x13, 11x11, 7x7 and 5x5 -->
            11x11 in the 1st and 7x7 in the second layer was the winner
            - Number of conv 256,384 and 512 capture features
            from more dimensions so the higher the better (512 was the winner)
        :return: The model and and convolution array
        '''
        print("Building PlanktonNet")
        print(self.name)
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))
        # Define the network architecture
        print("Define the network architecture...")
        #network = input_data(shape=[None, 227, 227, 3])
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        net = conv_2d(net, 96, 13, strides=4, activation='relu', name='conv_1')
        conv_1 = net
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)
        net = conv_2d(net, 256, 7, padding='same', activation='relu', name='conv_2')
        conv_2 = net
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)
        net = conv_2d(net, 512, 3, activation='relu', name='conv_3')
        conv_3 = net
        net = conv_2d(net, 512, 3, activation='relu', name='conv_4')
        conv_4 = net
        net = conv_2d(net, 512, 3, activation='relu', name='conv_5')
        conv_5 = net
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)
        net = conv_2d(net, 512, 3, activation='relu', name='conv_3')
        conv_3 = net
        net = conv_2d(net, 512, 3, activation='relu', name='conv_4')
        conv_4 = net
        net = conv_2d(net, 512, 3, activation='relu', name='conv_5')
        conv_5 = net
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)
        net = fully_connected(net, 4096, activation='tanh')
        net = dropout(net, self.keep_prob)
        net = fully_connected(net, 4096, activation='tanh')
        net = dropout(net, self.keep_prob)
        net = fully_connected(net, self.num_classes + 1, activation='softmax')
        net = regression(net, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=self.learning_rate, name='target')
        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2, conv_3, conv_4, conv_5]
        return model, conv_arr

    def __build_AlexNet(self):
        '''
        Build the model
        AlexNet - the version of AlexNet reconstructed to be compatible with the input images
        :return: The model and and convolution array
        '''
        print("Building AlexNet")
        print(self.name)
        #tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))
        # Define the network architecture
        print("Define the network architecture...")
        #network = input_data(shape=[None, 227, 227, 3])
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        net = conv_2d(net, 96, 11, strides=4, activation='relu', name='conv_1')
        conv_1 = net
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)
        net = conv_2d(net, 256, 5, activation='relu', name='conv_2')
        conv_2 = net
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)
        net = conv_2d(net, 384, 3, activation='relu', name='conv_3')
        conv_3 = net
        net = conv_2d(net, 384, 3, activation='relu', name='conv_4')
        conv_4 = net
        net = conv_2d(net, 256, 3, activation='relu', name='conv_5')
        conv_5 = net
        net = max_pool_2d(net, 3, strides=2)
        net = local_response_normalization(net)
        net = fully_connected(net, 4096, activation='tanh')
        net = dropout(net, self.keep_prob)
        net = fully_connected(net, 4096, activation='tanh')
        net = dropout(net, self.keep_prob)
        net = fully_connected(net, self.num_classes + 1, activation='softmax')
        net = regression(net, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=self.learning_rate, name='target')
        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2, conv_3, conv_4, conv_5]
        return model, conv_arr

    def __build_VGGNet(self):
        '''
        Build the model
        VGGNet - the version of VGGNet reconstructed to be compatible with the input images
        :return: The model and and convolution array
        '''
        print("Building" + self.name + " model ...")
        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        print("Block 1: 2 convolution layers each with 64 filters (each 3x3x3) followed by max_pool...")
        net = conv_2d(net, 64, 3, activation='relu',name='conv_1')
        conv_1 = net
        net = conv_2d(net, 64, 3, activation='relu',name='conv_2')
        conv_2 = net
        net = max_pool_2d(net, 2, strides=2)

        print("Block 2: 2 convolution layers each with 128 filters (each 3x3x3) followed by max_pool...")
        net = conv_2d(net, 128, 3, activation='relu',name='conv_3')
        conv_3 = net
        net = conv_2d(net, 128, 3, activation='relu',name='conv_4')
        conv_4 = net
        net = max_pool_2d(net, 2, strides=2)

        print("Block 3: 3 convolution layers each with 256 filters (each 3x3x3) followed by max_pool...")
        net = conv_2d(net, 256, 3, activation='relu', name='conv_5')
        conv_5 = net
        net = conv_2d(net, 256, 3, activation='relu', name='conv_6')
        conv_6 = net
        net = conv_2d(net, 256, 3, activation='relu', name='conv_7')
        conv_7 = net
        net = max_pool_2d(net, 2, strides=2)

        print("Block 4: 3 convolution layers each with 512 filters (each 3x3x3) followed by max_pool...")
        net = conv_2d(net, 512, 3, activation='relu', name='conv_8')
        conv_8 = net
        net = conv_2d(net, 512, 3, activation='relu', name='conv_9')
        conv_9 = net
        net = conv_2d(net, 512, 3, activation='relu', name='conv_10')
        conv_10 = net
        net = max_pool_2d(net, 2, strides=2)

        print("Block 5: 3 convolution layers each with 512 filters (each 3x3x3) followed by max_pool...")
        net = conv_2d(net, 512, 3, activation='relu', name='conv_11')
        conv_11 = net
        net = conv_2d(net, 512, 3, activation='relu', name='conv_12')
        conv_12 = net
        net = conv_2d(net, 512, 3, activation='relu', name='conv_13')
        conv_13 = net
        net = max_pool_2d(net, 2, strides=2)

        net = fully_connected(net, 4096, activation='relu')
        net = dropout(net, self.keep_prob)
        net = fully_connected(net, 4096, activation='relu')
        net = dropout(net, self.keep_prob)
        net = fully_connected(net, self.num_classes+1, activation='softmax')

        net = regression(net, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=self.learning_rate, name='target')

        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7,
                    conv_8, conv_9, conv_10, conv_11, conv_12, conv_13]
        return model, conv_arr

    def __build_GoogLeNet(self):
        '''
        Build the model
        GoogleNet (Inception Net) - the version of GoogleNet reconstructed to be compatible with the input images
        :return: The model and and convolution array
        '''
        print("Building GoogLeNet")
        print(self.name)
        print("Building" + self.name + " model ...")
        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        network = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                         data_preprocessing=self.__preprocessing(),
                         data_augmentation=self.__data_augmentation(), name='input')
        conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
        pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
        pool1_3_3 = local_response_normalization(pool1_3_3)
        conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
        conv2_3_3 = local_response_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
        conv_1 = conv1_7_7
        conv_2 = conv2_3_3


        # 3a
        print(" 3a ")
        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu',
                                   name='inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_3a_5_5_reduce')
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu',
                                   name='inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
        inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu',
                                        name='inception_3a_pool_1_1')
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                    mode='concat', axis=3)
        conv_3 = inception_3a_pool_1_1

        # 3b
        print(" 3b ")
        inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu',
                                          name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu',
                                   name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu',
                                          name='inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, name='inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_3b_pool_1_1')
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                    mode='concat', axis=3, name='inception_3b_output')
        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
        conv_4 = inception_3b_pool_1_1

        # 4a
        print(" 4a ")
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu',
                                          name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu',
                                   name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu',
                                   name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1, name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4a_pool_1_1')
        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                    mode='concat', axis=3, name='inception_4a_output')
        conv_5 = inception_4a_pool_1_1

        # 4b
        print(" 4b ")
        inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu',
                                          name='inception_4b_3_3_reduce')
        inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu',
                                   name='inception_4b_3_3')
        inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu',
                                          name='inception_4b_5_5_reduce')
        inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4b_5_5')
        inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1, name='inception_4b_pool')
        inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4b_pool_1_1')
        inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                    mode='concat', axis=3, name='inception_4b_output')
        conv_6 = inception_4b_pool_1_1

        # 4c
        print(" 4c ")
        inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
        inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',
                                          name='inception_4c_3_3_reduce')
        inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu',
                                   name='inception_4c_3_3')
        inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu',
                                          name='inception_4c_5_5_reduce')
        inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4c_5_5')
        inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
        inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4c_pool_1_1')
        inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                    mode='concat', axis=3, name='inception_4c_output')
        conv_7 = inception_4c_pool_1_1

        # 4d
        print(" 4d ")
        inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
        inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu',
                                          name='inception_4d_3_3_reduce')
        inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu',
                                   name='inception_4d_3_3')
        inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu',
                                          name='inception_4d_5_5_reduce')
        inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4d_5_5')
        inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1, name='inception_4d_pool')
        inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4d_pool_1_1')
        inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                    mode='concat', axis=3, name='inception_4d_output')
        conv_8 = inception_4d_pool_1_1

        # 4e
        print(" 4e ")
        inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
        inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu',
                                          name='inception_4e_3_3_reduce')
        inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_4e_3_3')
        inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu',
                                          name='inception_4e_5_5_reduce')
        inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_4e_5_5')
        inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1, name='inception_4e_pool')
        inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu',
                                        name='inception_4e_pool_1_1')
        inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1],
                                    axis=3, mode='concat')
        pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')
        conv_9 = inception_4e_pool_1_1

        # 5a
        print(" 5a ")
        inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
        inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu',
                                          name='inception_5a_3_3_reduce')
        inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_5a_3_3')
        inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu',
                                          name='inception_5a_5_5_reduce')
        inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5a_5_5')
        inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1, name='inception_5a_pool')
        inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5a_pool_1_1')
        inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],
                                    axis=3, mode='concat')
        conv_10 = inception_5a_pool_1_1

        # 5b
        print(" 5b ")
        inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
        inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu',
                                          name='inception_5b_3_3_reduce')
        inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu',
                                   name='inception_5b_3_3')
        inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu',
                                          name='inception_5b_5_5_reduce')
        inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5b_5_5')
        inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1, name='inception_5b_pool')
        inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5b_pool_1_1')
        inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],
                                    axis=3, mode='concat')
        pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
        pool5_7_7 = dropout(pool5_7_7, self.keep_prob)

        conv_11 = inception_5b_pool_1_1

        # fc
        print(" fc ")
        loss = fully_connected(pool5_7_7, self.num_classes+1, activation='softmax')
        network = regression(loss, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=self.learning_rate)


        # Wrap the network in a model object
        model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path=self.check_point_file)

        conv_arr = [conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7,
                    conv_8, conv_9, conv_10, conv_11]
        return model, conv_arr

    def __build_ResNet(self):
        '''
        Build the model
        ResNet (Residual Network) - the version of ResNet reconstructed to be compatible with the input images
        :return: The model and and convolution array
        '''
        # Residual blocks
        # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
        n = 5
        print("Building ResNet")
        print(self.name)
        print("Building" + self.name + " model ...")
        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                             data_preprocessing=self.__preprocessing(),
                             data_augmentation=self.__data_augmentation(), name='input')

        net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001, name='conv_1')
        conv_1 = net
        net = tflearn.residual_block(net, n, 16)
        net = tflearn.residual_block(net, 1, 32, downsample=True)
        net = tflearn.residual_block(net, n - 1, 32)
        net = tflearn.residual_block(net, 1, 64, downsample=True)
        net = tflearn.residual_block(net, n - 1, 64)
        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)
        # Regression
        net = tflearn.fully_connected(net, self.num_classes+1, activation='softmax')
        mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
        net = tflearn.regression(net, optimizer=mom,
                                 loss='categorical_crossentropy')

        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path=self.check_point_file, clip_gradients=0.)
        conv_arr = [conv_1]
        return model, conv_arr

    def __build_ResNeXt(self):
        '''
        Build the model
        ResNeXt - Aggregated residual transformations network (ResNeXt)
        :return: The model and and convolution array
        '''
        print("Building ResNeXt")
        print(self.name)
        print("Building" + self.name + " model ...")
        # Residual blocks
        # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
        n = 5

        # This resets all parameters and variables, leave this here
        tf.reset_default_graph()
        # Include the input layer, hidden layer(s), and set how you want to train the model
        inputsize = self.input_width * self.input_height * self.input_channels
        print("Inputlayer-size: %d" % (inputsize))

        # Define the network architecture
        print("Define the network architecture...")
        net = input_data(shape=[None, self.input_width, self.input_height, self.input_channels],
                             data_preprocessing=self.__preprocessing(),
                             data_augmentation=self.__data_augmentation(), name='input')
        net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001, name='conv_1')
        conv_1 = net
        net = tflearn.resnext_block(net, n, 16, 32)
        net = tflearn.resnext_block(net, 1, 32, 32, downsample=True, downsample_strides=1)
        net = tflearn.resnext_block(net, n - 1, 32, 32)
        net = tflearn.resnext_block(net, 1, 64, 32, downsample=True, downsample_strides=1)
        net = tflearn.resnext_block(net, n - 1, 64, 32)
        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)
        # Regression
        net = tflearn.fully_connected(net, self.num_classes+1, activation='softmax')
        opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
        net = tflearn.regression(net, optimizer=opt,
                                 loss='categorical_crossentropy')

        # Wrap the network in a model object
        model = tflearn.DNN(net, tensorboard_verbose=3,
                            checkpoint_path=self.check_point_file,
                            clip_gradients=0.)
        conv_arr = [conv_1]
        return model, conv_arr