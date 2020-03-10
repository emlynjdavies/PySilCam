# Import tflearn and some helpers
import os
import numpy as np
from skimage import exposure
import tensorflow as tf


from tflearn.data_utils import image_preloader  # shuffle,
from tflearn.data_utils import shuffle
from statistics import mean,stdev
#from pysilcam.net import Net
import h5py
import tflearn

import skimage.io
import skimage.transform
import pandas as pd

import pysilcam.silcam_classify as sccl
import cv2
# -----------------------------
IMXY = 64
DATABASE_PATH = '/mnt/DATA/silcam_classification_database' # '/mnt/DATA/dbIII' #
#DATABASE_PATH = 'C:/Users/ayas/Projects/AILARON/db'
#DATABASE_PATH = 'Z:/DATA/dbIII'
MODEL_PATH = '/mnt/DATA/model/COAPNetDBI' #'/mnt/DATA/model/COAPNetDBIII'
#MODEL_PATH = 'C:/Users/ayas/Projects/AILARON/model/COAPNetTest'
LOG_FILE = os.path.join(MODEL_PATH, 'CoapNet.out')

# -----------------------------
def find_classes(d=DATABASE_PATH):
    classes = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    print(classes)
    return classes


def add_im_to_stack(stack,im):
    blank = np.zeros([1, IMXY, IMXY, 3],dtype='uint8')
    # 1. normalize
    nmin = 0
    nmax = 255
    im = cv2.normalize(im, None, alpha=nmin, beta=nmax, norm_type=cv2.NORM_MINMAX)
    # 2. blur
    # 2. blur
    im = im.astype('float32')
    try:
        # Use blur_limit 5 instead of default 7. So this augmentation will
        # apply `cv2.medianBlur` using ksize=3 or ksize=5.
        im = cv2.medianBlur(blur_limit=3)(image=im).get('image')
    except Exception:
        fail = 1
        img = im
    im = cv2.medianBlur(im, 3)
    imrs = skimage.transform.resize(im, (IMXY, IMXY, 3), mode='reflect',
            preserve_range=True)
    # Contrast stretching
    #p2, p98 = np.percentile(imrs, (2, 98))
    #imrs = exposure.rescale_intensity(imrs, in_range=(p2, p98))
    #imrs = np.uint8(imrs)
            
    stack = np.vstack((stack, blank))
    stack[-1,:] = imrs
    return stack


def add_class_to_stack(stack,classes,classification_index):
    tag = np.zeros((1,len(classes)),dtype='uint8')
    tag[0][classification_index] = 1
    stack = np.vstack((stack, tag[0]))
    return stack

# -----------------------------
print('Formatting database....')
classes = find_classes()

X = np.zeros([0, IMXY, IMXY, 3],dtype='uint8')
Y = np.zeros((0,len(classes)),dtype='uint8')

for c_ind, c in enumerate(classes):
    print('  ',c)
    filepath = os.path.join(DATABASE_PATH,c)
    #files = os.listdir(filepath)
    files = [o for o in os.listdir(filepath) if o.endswith('.tiff')]
    for f in files:
        im = skimage.io.imread(os.path.join(filepath,f))
        X = add_im_to_stack(X, im)
        Y = add_class_to_stack(Y, classes, c_ind)

print('  Done.')

print('Splitting validation and training data')
print('70:15:15 - training, validation and test')

print('Total shape:', np.shape(Y), np.shape(X))

X, Y = shuffle(X, Y)
print('Total number of items ', X.shape[0])
print('number of items for the training set ', int(X.shape[0]*0.70))
print('number of items for the validation set ', int(X.shape[0]*0.15))

X_val = np.zeros([0, IMXY, IMXY, 3],dtype='uint8')
Y_val = np.zeros((0,len(classes)),dtype='uint8')

X_test = np.zeros([0, IMXY, IMXY, 3],dtype='uint8')
Y_test = np.zeros((0,len(classes)),dtype='uint8')

for c in range(len(classes)):
    ind = np.argwhere(Y[:,c]==1)
    print(len(ind),'images in class',c)
    #step = np.max([int(np.round(len(ind)/10)),1])
    step = int(len(ind)*0.15/len(classes))  # X.shape[0]
    step = max(step, 1)
    print('  to be shortened by', len(ind)*0.15/len(classes))
    #ind = np.array(ind[0::step]).flatten()
    ind1 = np.array(ind[np.arange(0, step)]).flatten()
    print('ind ', ind)
    print('len(ind) ', len(ind))
    print('ind1 ', ind1)
    print('len(ind1) ', len(ind1))

    Y_test = np.vstack((Y_test,Y[ind1,:]))
    X_test = np.vstack((X_test,X[ind1,:,:,:]))
    print('  test shape:', np.shape(Y_test), np.shape(X_test))

    ind2 = np.array(ind[np.arange(step, 2*step)]).flatten()
    print('ind2 ', ind2)
    print('len(ind2) ', len(ind2))

    Y_val = np.vstack((Y_val, Y[ind2, :]))
    X_val = np.vstack((X_val, X[ind2, :, :, :]))
    print('  validation set shape:', np.shape(Y_val), np.shape(X_val))

    Y = np.delete(Y,ind1,0)
    Y = np.delete(Y, ind2, 0)
    X = np.delete(X,ind1,0)
    X = np.delete(X, ind2, 0)
    print('  data shape:', np.shape(Y), np.shape(X))

print('OK.')

# -----------------------------
df = pd.DataFrame(columns = classes)
header_file = os.path.join(MODEL_PATH, 'header.tfl.txt')
df.to_csv(header_file, index=False)
# -----------------------------
outputs = np.shape(Y)[1]

X = np.float64(X)
Y = np.float64(Y)
X_test = np.float64(X_test)
Y_test = np.float64(Y_test)
X_val = np.float64(X_val)
Y_val = np.float64(Y_val)

print('Shuffle the data')
X, Y = shuffle(X, Y)

# ------------------------------------

n_epoch = 200
batch_size = 128

model_file = os.path.join(MODEL_PATH, 'plankton-classifier.tfl')
print('MODEL_PATH ', MODEL_PATH)
print('model_file ', model_file)
model, conv_arr = sccl.build_model(model_file) #myNet.build_model(model_file)
print('model', model)
model_name = MODEL_PATH + '/plankton-classifier'
print('model_name ', model_name)
#myNet.train(model, X, Y, X_test, Y_test, '', n_epoch, batch_size, model_name=model_name)
sccl.train_model(X, Y, X_val, Y_val, batch_size, n_epoch, model)
# Save
print("Saving model ...")
model.save('plankton-classifier.tfl')
# Predict
print('PREDICT')
model.load('plankton-classifier.tfl')
prediction = sccl.predict(X_test[3], model)
print('Prediction ', prediction)
print('True ', Y_test[3])
# Evaluate
print('NETWORK EVALUATION')
y_pred, y_true, accuracy, precision, recall, f1_score, confusion_matrix, normalized_confusion_matrix = \
    sccl.evaluate_model(model, X_test, Y_test)
print("\nAccuracy: {}%".format(100*accuracy))
print("Precision: {}%".format(100 * precision))
print("Recall: {}%".format(100 * recall))
print("F1_Score: {}%".format(100 * f1_score))
print("confusion_matrix: ", confusion_matrix)
print("Normalized_confusion_matrix: ", normalized_confusion_matrix)

print("Network trained and saved as particle-classifier.tfl!")
