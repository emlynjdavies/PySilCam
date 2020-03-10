  # -*- coding: utf-8 -*-
import tensorflow as tf
import scipy
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from skimage import exposure
from pysilcam.net import Net,CoapNet
import cv2

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

def load_model(model_path='/mnt/ARRAY/classifier/model/plankton-classifier.tfl'):

    model, class_labels = build_model(model_path)
    model.load(model_path)

    return model, class_labels

def build_model(model_path='/mnt/ARRAY/classifier/model/plankton-classifier.tfl'):
    path, filename = os.path.split(model_path)
    print('path ',path)
    print('model_path ', model_path)
    print('filename', filename)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    OUTPUTS = len(header.columns)
    class_labels = header.columns

    name = 'CoapNet'
    input_width = 64
    input_height = 64
    input_channels = 3
    num_classes = OUTPUTS

    learning_rate = 0.001
    #myNet = Net(name, input_width, input_height, input_channels, num_classes, learning_rate)
    myNet = CoapNet(name, input_width=input_width, input_height=input_height, input_channels=input_channels,
                      num_classes=num_classes, learning_rate=learning_rate)
    tf.reset_default_graph()
    #model_file = os.path.join(MODEL_PATH, '/plankton-classifier.tfl')
    model, conv_arr = myNet.build_model(model_path)

    return model, class_labels

def train_model(X, Y, X_val, Y_val, batch_size, n_epoch, model):
    model.fit(X, Y, n_epoch, shuffle=True, validation_set=(X_val, Y_val),
              show_metric=True, batch_size=batch_size,
              snapshot_epoch=True,
              run_id='plankton-classifier')
    return model


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
    # 1. normalize
    nmin = 0
    nmax = 255
    img = cv2.normalize(img, None, alpha=nmin, beta=nmax, norm_type=cv2.NORM_MINMAX)
    # 2. blur
    img = img.astype('float32')
    try:
        # Use blur_limit 5 instead of default 7. So this augmentation will
        # apply `cv2.medianBlur` using ksize=3 or ksize=5.
        img = cv2.medianBlur(blur_limit=3)(image=img).get('image')
    except Exception:
        fail = 1
        img = img
    #img = cv2.medianBlur(img, 3)

    # 3. Scale it to 32x32
    #img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
    img = scipy.misc.imresize(img, (64, 64), interp="bicubic").astype(np.float32, casting='unsafe')
    # Contrast stretching
    #p2, p98 = np.percentile(img, (2, 98))
    #img = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Predict
    prediction = model.predict([img])

    return prediction


def evaluate_model(model, X_test, Y_test):
    y_pred = []
    y_true = []
    pred = model.predict(X_test)
    for ty in pred:
        #print("ty, y_pred: ", ty, ty.argmax(axis=0))
        y_pred.append(ty.argmax(axis=0))
    for ty in Y_test:
        #print("ty, y_true: ", ty, ty.argmax(axis=0))
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

    print("\nAccuracy: {}%".format(100*accuracy))
    print("Precision: {}%".format(100 * precision))
    print("Recall: {}%".format(100 * recall))
    print("F1_Score: {}%".format(100 * f1_score))
    print("confusion_matrix: ", confusion_matrix)
    print("Normalized_confusion_matrix: ", normalized_confusion_matrix)
    # ------------------------------------
    return y_pred, y_true, accuracy, precision, recall, f1_score, confusion_matrix, normalized_confusion_matrix
