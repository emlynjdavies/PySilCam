import os
import tensorflow as tf
from tflearn.data_utils import image_preloader  # shuffle,
from statistics import mean,stdev
from make_data import MakeData
from net import Net



# -- PATHS ---------------------------
# DATABASE_PATH = 'Z:/DATA/dataset_test'
# MODEL_PATH = 'Z:/DATA/model/modelCV2'
DATABASE_PATH = '/mnt/DATA/silcam_classification_database'
#DATABASE_PATH = '/mnt/DATA/dataset'
MODEL_PATH = '/mnt/DATA/model/modelMINST'
#DATABASE_PATH = 'Z:/DATA/dataset'
#MODEL_PATH = 'Z:/DATA/model/modelORGNET'
LOG_FILE = os.path.join(MODEL_PATH, 'MINSTDNNGPUSMALL.log')
HEADER_FILE = os.path.join(MODEL_PATH, "header.tfl.txt")         # the header file that contains the list of classes
set_file = os.path.join(DATABASE_PATH,"image_set.dat")     # the file that contains the list of images of the testing dataset along with their classes
# set_file = os.path.join(MODEL_PATH,"image_set_win.dat")     # the file that contains the list of images of the testing dataset along with their classes

# -----------------------------
SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset

name='MINST'
input_width=32 # input_width=224
input_height=32 # input_height=224
input_channels=3
num_classes=7

learning_rate=0.01 # 0.01 for LeNet -- 0.001 for OrgNet -- 0.01 for MINST -- 0.001 for CIFAR10 -- 0.001 for AlexNet
                        # 0.0001 for VGGNet
momentum=0.9
keep_prob=0.8  # 0.75 for OrgNet -- 0.8 for LeNet -- 0.8 for MINST -- 0.5 for CIFAR10 -- 0.5 for AlexNet
                # 0.5 for VGGNET

n_epoch = 50  # 50
batch_size = 128 # 128

print('Call image_preloader ....')
X, Y = image_preloader(set_file, image_shape=(input_width, input_height, input_channels), mode='file', categorical_labels=True, normalize=True)
n_splits = 0
data_set = MakeData(X, Y, n_splits)

LeNet = Net(name, input_width, input_height, input_channels, num_classes, learning_rate,
                momentum, keep_prob)
fh = open(LOG_FILE, 'w')
trainX, testX, trainY, testY = data_set.makeXY(SPLIT_PERCENT)

tf.reset_default_graph()
print("trainY: ", trainY)
print("testY: ", testY)

model_file = os.path.join(MODEL_PATH, name +'GPUSMALL/plankton-classifier.tfl')
model, conv_arr = LeNet.build_model(model_file)

# Training
print("start training for NN  ...", name)
LeNet.train(model, trainX, trainY, testX, testY, name, n_epoch, batch_size)

# Save
print("Saving model ...", name)
model.save(model_file)

# Evaluate
y_pred, y_true, accuracy, precision, recall, f1_score, confusion_matrix, normalised_confusion_matrix = \
    LeNet.evaluate(model, testX, testY)

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
