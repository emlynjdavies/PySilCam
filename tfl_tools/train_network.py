# Import tflearn and some helpers
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, image_preloader
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import numpy as np
import skimage.io
import skimage.transform
import os
import pandas as pd
import math
import pysilcam.silcam_classify as sccl
# -----------------------------
DATABASE_PATH = 'Z:/DATA/silcam_classification_database'
MODEL_PATH = 'Z:/DATA/model/model005'
HEADER_FILE = os.path.join(MODEL_PATH, "header.tfl.txt")         # the header file that contains the list of classes
trainset_file = os.path.join(MODEL_PATH,"imagelist_train.dat")   # the file that contains the list of images of the training dataset along with their classes
testset_file = os.path.join(MODEL_PATH,"imagelist_test.dat")     # the file that contains the list of images of the testing dataset along with their classes
IMXY = 32
SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset
CHECK_POINT_FILE = "plankton-classifier.tfl.ckpt"
MODEL_FILE = "plankton-classifier.tfl"

# --- FUNCTION DEFINITION --------------------------
def find_classes(d=DATABASE_PATH):
    classes = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    print(classes)
    return classes

def save_classes(classList):
    df_classes = pd.DataFrame(columns=classList)
    df_classes.to_csv(HEADER_FILE, index=False)

# --- get file list from the folder structure
def import_directory_structure(classList):
    fileList = []
    for c_ind, c in enumerate(classList):
        print('  ', c)
        filepath = os.path.join(DATABASE_PATH, c)
        files = [o for o in os.listdir(filepath) if o.endswith('.tiff')]
        for f in files:
            fileList.append([os.path.join(filepath, f), str(c_ind + 1)])
    fileList = np.array(fileList)
    return fileList

# -----------------------------
print('=== Formatting database....')
classList = find_classes()
save_classes(classList)
print("CLASSLIST SIZE ", pd.read_csv(HEADER_FILE, header=None).shape[1])
# --- get file list from the folder structure
print('Import directory structure....')
fileList = import_directory_structure(classList)
# -- shuffle the dataset
print('Shuffle dataset....')
np.random.shuffle(fileList)
# -- split into train and test data and save to files
print('Split into training and test datasets....')
split = math.floor(fileList.shape[0]*SPLIT_PERCENT)
print('Splitting %d dataset for test and the %d for train ' % (split, fileList.shape[0]-split))
test, train = fileList[:split,:],fileList[split:,:]
print('Save into test and train files ....')
np.savetxt(testset_file, test, delimiter=' ', fmt='%s')
np.savetxt(trainset_file, train, delimiter=' ', fmt='%s')

# -- call image_preloader
print('Call image_preloader ....')
testX, testY = image_preloader(testset_file, image_shape=(IMXY, IMXY, 3),   mode='file', categorical_labels=True, normalize=True)
trainX, trainY = image_preloader(trainset_file, image_shape=(IMXY, IMXY, 3),   mode='file', categorical_labels=True, normalize=True)

# -- print a sample
select= 300
print(trainY[select])
print(trainY[select].argmax(axis=0))

print(testY[select])
print(testY[select].argmax(axis=0))

# Build the model
print("MODEL_PATH ", MODEL_PATH, CHECK_POINT_FILE)

model, conv_arr, class_labels = sccl.build_model(IMXY, MODEL_PATH, CHECK_POINT_FILE)
# Training
print("start training ...")
model.fit(trainX, trainY, n_epoch=200, shuffle=True, validation_set=(testX, testY),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='plankton-classifier')
# Save
print("Saving ...")
model_file = os.path.join(MODEL_PATH,MODEL_FILE)
model.save(model_file)

# Evaluate model
score = model.evaluate(testX, testY)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))

# Run the model on one example
prediction = model.predict([testX[0]])
print("Prediction: %s" % str(prediction[0]))


