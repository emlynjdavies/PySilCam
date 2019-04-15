from __future__ import absolute_import, division, print_function
# Evaluate using Cross Validation
import os
import pandas as pd
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, image_preloader
import pysilcam.silcam_classify as sccl


from sklearn import model_selection, metrics
import tensorflow as tf
import numpy as np
from statistics import mean,stdev


# -- PATHS ---------------------------
DATABASE_PATH = '/mnt/DATA/dataset'
MODEL_PATH = '/mnt/DATA/model/modelCV'
HEADER_FILE = os.path.join(MODEL_PATH, "header.tfl.txt")         # the header file that contains the list of classes
set_file = os.path.join(MODEL_PATH,"image_set.dat")     # the file that contains the list of images of the testing dataset along with their classes
IMXY = 32
#SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset
# -----------------------------

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

def make_dataset(X_data,y_data,n_splits):
    seed = 7
    for train_index, test_index in model_selection.KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        yield X_train,y_train,X_test,y_test


# -----------------------------
# -----------------------------
'''print('=== Formatting database....')
classList = find_classes()
save_classes(classList)
print("CLASSLIST SIZE ", pd.read_csv(HEADER_FILE, header=None).shape[1])
# --- get file list from the folder structure
print('Import directory structure....')
fileList = import_directory_structure(classList)
# -- shuffle the dataset
print('Shuffle dataset....')
np.random.shuffle(fileList)

print('Save into a file ....')
np.savetxt(set_file, fileList, delimiter=' ', fmt='%s')'''
# -- call image_preloader
print('Call image_preloader ....')
X, Y = image_preloader(set_file, image_shape=(IMXY, IMXY, 3), mode='file', categorical_labels=True, normalize=True)
print(X[0])
print(Y[0])
i = 0
prediction = []
test = []
accuracy = []
precision = []
recall = []
f1_score = []
confusion_matrix = []
normalised_confusion_matrix = []
fh = open('/mnt/DATA/model/modelCV/cv.out', 'w')
for trainX, trainY, testX, testY in make_dataset(X, Y, 10):
    i = i + 1
    tf.reset_default_graph()
    round_num = str(i)
    if i < 10:
        round_num = '0' + round_num    # kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)

    CHECK_POINT_FILE = 'round'+ round_num + '/plankton-classifier.tfl.ckpt'
    MODEL_FILE = 'round'+ round_num + '/plankton-classifier.tfl'
    model_file = os.path.join(MODEL_PATH, MODEL_FILE)

    print("MODEL_PATH ", MODEL_PATH)
    print("CHECK_POINT_FILE ", CHECK_POINT_FILE)
    print("model_file ", model_file)
    model, conv_arr, class_labels = sccl.build_model(IMXY, MODEL_PATH, CHECK_POINT_FILE)
    # Training
    print("start training round %f ...", i)
    model.fit(trainX, trainY, n_epoch=50, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=128,
              snapshot_epoch=True,
              run_id='plankton-classifier'+round_num)
    # Save
    print("Saving model %f ..." % i)
    model.save(model_file)
    # Evaluate model
    # score = model.evaluate(testX, testY)
    # print('Test accuracy: %0.4f%%' % (score[0] * 100))
    # score = model.evaluate(testX, testY)
    # fh.write("Accuracy for round %f: %.4f%% " % i, (score[0] * 100))

    # print("\nTest prediction for x = ", testX)
    print("model evaluation ")
    predictions = model.predict(testX)
    #predictions = [int(i) for i in model.predict(testX)]
    print("predictions: ", predictions)
    fh.write("\npredictions: ")
    y_pred = []
    for pred in predictions:
        y_pred.append(np.argmax(pred))
    print(y_pred)
    for el in y_pred:
        fh.write("%s " % el)
    print("testY: ")
    fh.write("\ntestY: ")
    y_true = []
    for ty in testY:
        y_true.append(ty.argmax(axis=0))
    print(y_true)
    for el in y_true:
        fh.write("%s " % el)

    acc = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy: {}%".format(100 * acc))
    fh.write("\nAccruacy: {}%".format(100 * acc))

    #print("testY: ", testY)
    pre = metrics.precision_score(y_true, y_pred, average="weighted")
    print("Precision: {}%".format(100 * pre))
    fh.write("\tPrecision: {}%".format(100 * pre))

    rec = metrics.recall_score(y_true, y_pred, average="weighted")
    print("Recall: {}%".format(100 * rec))
    fh.write("\tRecall: {}%".format(100 * rec))

    f1sc = metrics.f1_score(y_true, y_pred, average="weighted")
    print("f1_score: {}%".format(100 * f1sc))
    fh.write("\tf1_score: {}%".format(100 * f1sc))
    print("")
    print("Confusion Matrix:")
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    norm_conf_matrix = np.array(conf_matrix, dtype=np.float32) / np.sum(conf_matrix) * 100
    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(norm_conf_matrix)

    ## update summaries ###
    prediction.append(predictions)
    test.append(testY)
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    f1_score.append(f1sc)
    confusion_matrix.append(conf_matrix)
    normalised_confusion_matrix.append(norm_conf_matrix)
    

for i in range(0, 10):
    print("Round ", i)
    print("Accuracy: {}%".format(100*accuracy[i]))
    print("Precision: {}%".format(100 * precision[i]))
    print("Recall: {}%".format(100 * recall[i]))
    print("F1 Score: {}%".format(100 * f1_score[i]))
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
