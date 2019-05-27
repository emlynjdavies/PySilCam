import os
import tensorflow as tf
import h5py
from tflearn.data_utils import image_preloader  # shuffle,
from statistics import mean,stdev
from make_data import MakeData
from net import Net
from tflearn.data_utils import build_hdf5_image_dataset



# -- PATHS ---------------------------
#DATABASE_PATH = 'Z:/DATA/dataset_test'
#MODEL_PATH = 'Z:/DATA/model/modelCV2'
DATABASE_PATH = '/mnt/DATA/silcam_classification_database'
#DATABASE_PATH = '/mnt/DATA/dataset'
MODEL_PATH = '/mnt/DATA/model/modelVGGNET'
#DATABASE_PATH = 'Z:/DATA/dataset'
# DATABASE_PATH = 'Z:/DATA/silcam_classification_database'
# MODEL_PATH = 'Z:/DATA/model/modelVGGNET'
LOG_FILE = os.path.join(MODEL_PATH, 'VGGDNetGPUSMALL.log')
HEADER_FILE = os.path.join(MODEL_PATH, "header.tfl.txt")         # the header file that contains the list of classes
set_file = os.path.join(DATABASE_PATH,"image_set.dat")     # the file that contains the list of images of the testing dataset along with their classes
# set_file = os.path.join(DATABASE_PATH,"image_set_win.dat")     # the file that contains the list of images of the testing dataset along with their classes
out_hd5 = os.path.join(MODEL_PATH,"datasetsmall.h5")
# -----------------------------
SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset

name='VGGNet'
input_width=224
input_height=224
input_channels=3
num_classes=7

learning_rate=0.0001  # 0.001 for OrgNet -- 0.01 for MINST -- 0.001 for CIFAR10 -- 0.001 for AlexNet
                        # 0.0001 for VGGNet
momentum=0.9
keep_prob=0.5  # 0.75 for OrgNet -- 0.8 for LeNet -- 0.5 for CIFAR10 -- 0.5 for AlexNet
                # 0.5 for VGGNET

n_epoch = 50  # 50
batch_size = 128 # 128

#print('Call image_preloader ....')
#X, Y = image_preloader(set_file, image_shape=(input_width, input_height, input_channels),
#                       mode='file', categorical_labels=True, normalize=True)
#print('Call build hdf5 image dataset ....')
#build_hdf5_image_dataset(set_file, image_shape=(input_width, input_height, input_channels),
#                         mode='file', output_path=out_hd5, categorical_labels=True, normalize=True)
#h5f = h5py.File(out_hd5, 'r')
#X = h5f['X'].value
#Y = h5f['Y'].value

n_splits = 0
data_set = MakeData(n_splits= n_splits)
'''
data_set.create_hdf5(set_file, DATABASE_PATH,
                    input_width = input_width, input_height = input_height, input_channels = 3,
                    split_percent =0.05, win = '')
'''
win = '' # ''_win' when operating on windows environment
out_test_hd5 = os.path.join(DATABASE_PATH, 'image_set_test' + str(input_width) + win + ".h5")
out_train_hd5 = os.path.join(DATABASE_PATH, 'image_set_train' + str(input_width) + win + ".h5")
train_h5f = h5py.File(out_train_hd5, 'r')
test_h5f = h5py.File(out_test_hd5, 'r')
trainX = train_h5f['X']
trainY = train_h5f['Y']
testX = test_h5f['X']
testY = test_h5f['Y']

print(train_h5f['X'].shape)
print(train_h5f['Y'].shape)
print(test_h5f['X'].shape)
print(test_h5f['Y'].shape)


myNet = Net(name, input_width, input_height, input_channels, num_classes, learning_rate,
                momentum, keep_prob)
fh = open(LOG_FILE, 'w')
# trainX, testX, trainY, testY = data_set.makeXY(SPLIT_PERCENT)
tf.reset_default_graph()

model_file = os.path.join(MODEL_PATH, name +'GPUSMALL/plankton-classifier.tfl')
model, conv_arr = myNet.build_model(model_file)

# Training
print("start training for NN  ...", name)
myNet.train(model, trainX, trainY, testX, testY, name, n_epoch, batch_size)

# Save
print("Saving model ...", name)
model.save(model_file)

# Evaluate
y_pred, y_true, accuracy, precision, recall, f1_score, confusion_matrix, normalised_confusion_matrix = \
    myNet.evaluate(model, testX, testY)

fh.write("\nRound ")
fh.write("\nPredictions: ")
for el in y_pred:
    fh.write("%s " % el)
fh.write("\ny_true: ")
for el in y_true:
    fh.write("%s " % el)

print("\nAccuracy: {}%".format(100*accuracy))
fh.write("\nAccuracy: {}%".format(100*accuracy))
print("Precision: {}%".format(100 * precision))
fh.write("\tPrecision: {}%".format(100 * precision))
print("Recall: {}%".format(100 * recall))
fh.write("\tRecall: {}%".format(100 * recall))
print("F1_Score: {}%".format(100 * f1_score))
fh.write("\tF1_Score: {}%".format(100 * f1_score))
print("confusion_matrix: ", confusion_matrix)
print("Normalized_confusion_matrix: ", normalised_confusion_matrix)

fh.close
