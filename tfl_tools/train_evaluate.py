import os
import tensorflow as tf
from tflearn.data_utils import image_preloader  # shuffle,
from statistics import mean,stdev
from make_data import MakeData
from net import Net



# -- PATHS ---------------------------
DATABASE_PATH = 'Z:/DATA/dataset_test'
MODEL_PATH = 'Z:/DATA/model/modelCV2'
#DATABASE_PATH = '/mnt/DATA/dataset_test'
#MODEL_PATH = '/mnt/DATA/model/modelCV2'
LOG_FILE = os.path.join(MODEL_PATH, 'cv.out')
# DATABASE_PATH = '/mnt/DATA/dataset'
# MODEL_PATH = '/mnt/DATA/model/modelCV'
HEADER_FILE = os.path.join(MODEL_PATH, "header.tfl.txt")         # the header file that contains the list of classes
# set_file = os.path.join(MODEL_PATH,"image_set.dat")     # the file that contains the list of images of the testing dataset along with their classes
set_file = os.path.join(MODEL_PATH,"image_set_win.dat")     # the file that contains the list of images of the testing dataset along with their classes

IMXY = 32
# -----------------------------

name='LeNet'
input_width=32
input_height=32
input_channels=3
num_classes=7

learning_rate=0.01  # for LeNet -- 0.001 for CIFAR10
momentum=0.9
keep_prob=0.8  # for LeNet -- 0.5 for CIFAR10

n_epoch = 3  # 50
batch_size = 3 # 128

print('Call image_preloader ....')
X, Y = image_preloader(set_file, image_shape=(IMXY, IMXY, 3), mode='file', categorical_labels=True, normalize=True)
n_splits = 10
data_set = MakeData(X, Y, n_splits)

i = 0
prediction = []
test = []
accuracy = []
precision = []
recall = []
f1_score = []
confusion_matrix = []
normalised_confusion_matrix = []
LeNet = Net(name, input_width, input_height, input_channels, num_classes, learning_rate,
                momentum, keep_prob)
fh = open(LOG_FILE, 'w')
for trainX, trainY, testX, testY in data_set.gen():

    tf.reset_default_graph()
    i = i + 1
    round_num = str(i)
    if i < 10:
        round_num = '0' + round_num
    print("Round # ", round_num)
    print("trainY: ", trainY)
    print("testY: ", testY)

    model_file = os.path.join(MODEL_PATH, 'round' + round_num + '/plankton-classifier.tfl')
    model, conv_arr = LeNet.build_model(model_file)

    # Training
    print("start training round %f ...", i)
    LeNet.train(model, trainX, trainY, testX, testY, round_num, n_epoch, batch_size)

    # Save
    print("Saving model %f ..." % i)
    model.save(model_file)

    y_pred, y_true, acc, pre, rec, f1sc, conf_matrix, norm_conf_matrix = \
        LeNet.evaluate(model, testX, testY)

    ## update summaries ###
    prediction.append(y_pred)
    test.append(y_true)
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    f1_score.append(f1sc)
    confusion_matrix.append(conf_matrix)
    normalised_confusion_matrix.append(norm_conf_matrix)


for i in range(0, 10):
    print("Round ", i)
    fh.write("\nRound")
    fh.write(round_num)
    fh.write("\nPredictions: ")
    for el in y_pred:
        fh.write("%s " % el)
    fh.write("\ny_true: ")
    for el in y_true:
        fh.write("%s " % el)
    print("\nAccuracy: {}%".format(100*accuracy[i]))
    fh.write("\tAccuracy: {}%".format(100*accuracy[i]))
    print("Precision: {}%".format(100 * precision[i]))
    fh.write("\tPrecision: {}%".format(100 * precision[i]))
    print("Recall: {}%".format(100 * recall[i]))
    fh.write("\tRecall: {}%".format(100 * recall[i]))
    print("F1 Score: {}%".format(100 * f1_score[i]))
    fh.write("F1 Score: {}%".format(100 * f1_score[i]))
    print("confusion matrix: ", confusion_matrix[i])
    print("Normalized confusion matrix: ", normalised_confusion_matrix[i])

print("Overall Accuracy: %.3f%% (%.3f%%)" % (mean(accuracy)*100.0, stdev(accuracy)*100.0))
fh.write("\nOverall Accuracy: %.3f%% (%.3f%%)" % (mean(accuracy)*100.0, stdev(accuracy)*100.0))
print("Overall Precision: %.3f%% (%.3f%%)" % (mean(precision)*100.0, stdev(precision)*100.0))
fh.write("\tOverall Precision: %.3f%% (%.3f%%)" % (mean(precision)*100.0, stdev(precision)*100.0))
print("Overall Recall: %.3f%% (%.3f%%)" % (mean(recall)*100.0, stdev(recall)*100.0))
fh.write("\tOverall Recall: %.3f%% (%.3f%%)" % (mean(recall)*100.0, stdev(recall)*100.0))
print("Overall F1Score: %.3f%% (%.3f%%)" % (mean(f1_score)*100.0, stdev(f1_score)*100.0))
fh.write("\tOverall F1Score: %.3f%% (%.3f%%)" % (mean(f1_score)*100.0, stdev(f1_score)*100.0))
print('Confusion Matrix')
for i in range(0,10):
    print(confusion_matrix[i])

print('Normalized Confusion Matrix')
for i in range(0,10):
    print(normalised_confusion_matrix[i])

fh.close
