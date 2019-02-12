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
MODEL_PATH = 'Z:/DATA/model/model004'
HEADER_FILE = os.path.join(MODEL_PATH, "header.tfl.txt")         # the header file that contains the list of classes
trainset_file = os.path.join(MODEL_PATH,"imagelist_train.dat")   # the file that contains the list of images of the training dataset along with their classes
testset_file = os.path.join(MODEL_PATH,"imagelist_test.dat")     # the file that contains the list of images of the testing dataset along with their classes
IMXY = 32
SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset
CHECK_POINT_FILE = os.path.join(MODEL_PATH, "plankton-classifier.tfl.ckpt")
MODEL_FILE = os.path.join(MODEL_PATH, "plankton-classifier.tfl")
# --- FUNCTION DEFINITION --------------------------
def show_digit(index):
    label = testY[index].argmax(axis=0)
    image = testX[index]
    plt.title('Test data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image)
    plt.show()


# -----------------------------
print('Call image_preloader ....')
testX, testY = image_preloader(testset_file, image_shape=(IMXY, IMXY, 3),   mode='file', categorical_labels=True, normalize=True)
#trainX, trainY = image_preloader(trainset_file, image_shape=(IMXY, IMXY, 3),   mode='file', categorical_labels=True, normalize=True)

select= 300
print(testY[select])
print(testY[select].argmax(axis=0))

# Display the first (index 0) training image
#show_digit(29)
#show_digit(30)

# Build the model
print ('building model...')
model, conv_arr, class_labels = sccl.build_model(IMXY, MODEL_PATH, MODEL_FILE)
# Load Model
print("Loading model ...")
model.load(os.path.join(MODEL_PATH,MODEL_FILE))
print ('MODEL Loaded ', model)
print ('CONV_ARR ', conv_arr)
for c in conv_arr:
    print(c)
print("class_labels ",class_labels)
# Evaluate model
score = model.evaluate(testX, testY)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))

# Run the model on one example
prediction = model.predict([testX[30]])
print("Prediction: %s" % str(prediction[0]))


#Look into filters
# choose images & plot the first one
idx=29
mylabel = testY[idx].argmax(axis=0)
im = testX[idx:idx+1]
plt.axis('off')
plt.title('Image, index: %d,  Label: %d' % (idx, mylabel))
plt.imshow(im[0], cmap='gray_r')
plt.gcf().set_size_inches(1, 1)

# run images through 1st conv layer
m2 = tflearn.DNN(conv_arr[0], session=model.session)
yhat = m2.predict(im)

# slice off outputs for first image and plot
yhat_1 = np.array(yhat[0])

def vis_conv(v,ix,iy,ch,cy,cx, p = 0) :
    v = np.reshape(v,(iy,ix,ch))
    ix += 2
    iy += 2
    npad = ((1,1), (1,1), (0,0))
    v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
    v = np.reshape(v,(iy,ix,cy,cx))
    v = np.transpose(v,(2,0,3,1)) #cy,iy,cx,ix
    v = np.reshape(v,(cy*iy,cx*ix))
    return v

#  h_conv1 - processed image
ix = IMXY  # img size
iy = IMXY
ch = 32
cy = 4   # grid from channels:  32 = 4x8
cx = 8
#ch = 48
#cy = 6   # grid from channels:  32 = 4x8
#cx = 8

v  = vis_conv(yhat_1,ix,iy,ch,cy,cx)
myfig=plt.figure(figsize = (8,8))
plt.imshow(v,cmap="Greys_r",interpolation='nearest')
plt.axis('off');
fname="foo_"+str(idx)+".png"
myfig.savefig(fname)
plt.show()
## Acknowledgements @rgr on Stackoverflow, http://stackoverflow.com/a/35247876



print("start predicting ...")
#predict_y = model.predict(testX)
predict_y = model.predict(testX[idx:idx+1])
print(predict_y[0])
print(predict_y[0].argmax(axis=0))
print("Expected class: ")
print(testY[idx])
print(testY[idx].argmax(axis=0))
print("end predicting ...")

i=0;
failed=0
correct=0
for cnt in testX:
 predict_test = model.predict(testX[i:i+1])
 computedresult=predict_test[0].argmax(axis=0)
 expectedresult=testY[i].argmax(axis=0)
 if(computedresult != expectedresult):
     failed=failed+1;
     print("awk 'NR==",str(i+1),"' imagelist_test.dat   ",i,":",computedresult," GTruth=",expectedresult)
 if(computedresult == expectedresult):
     correct=correct+1;
 #print computedresult,expectedresult
 i=i+1

gesamt=failed+correct
rate=correct/gesamt
print("Accuracy:")
print("Success-rate:",rate)
print("correct:",correct,"failed:",failed)
print("Gesamt:",gesamt)

