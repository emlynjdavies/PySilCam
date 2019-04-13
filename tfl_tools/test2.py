""" Linear Regression Example """
from __future__ import absolute_import, division, print_function
from sklearn import model_selection, metrics
import tensorflow as tf
import tflearn
import numpy as np
from statistics import mean,stdev

def make_dataset(X_data, Y_data, n_splits):
    seed = 7
    for train_index, test_index in model_selection.KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(X_data):
        print('train_index test_index ' , train_index, test_index)
        X_train = []
        Y_train = []
        for train_el in train_index:
            X_train.append(X_data[train_el])
            Y_train.append(Y_data[train_el])
        X_test = []
        Y_test = []
        for test_el in test_index:
            X_test.append(X_data[test_el])
            Y_test.append(Y_data[test_el])
        yield np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)


# Regression data
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
#Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]
Y = [1,2,2,3,1,1,3,2,2,1,2,3,1,2,2,2,1]

i = 0
prediction = []
test = []
accuracy = []
precision = []
recall = []
f1_score = []
confusion_matrix = []
normalised_confusion_matrix = []

for trainX, trainY, testX, testY in make_dataset(X, Y, 10):
    i = i + 1
    tf.reset_default_graph()
    round_num = str(i)
    if i < 10:
        round_num = '0' + round_num

    input_ = tflearn.input_data(shape=[None])
    linear = tflearn.single_unit(input_)
    regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                    metric='R2', learning_rate=0.01)
    model = tflearn.DNN(regression)
    model.fit(trainX, trainY, n_epoch=3, validation_set=(testX,testY),
              show_metric=True, batch_size=3,
              snapshot_epoch=True,
              run_id='lin_reg'+round_num)
    score = model.evaluate(testX, testY)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))
    #accuracy.append(score[0]*100)

    print("\nTest prediction for x = ",testX)
    print("model evaluation ")
    predictions = model.predict(testX)
    predictions = [int(i) for i in model.predict(testX)]
    print("predictions: " , predictions)
    print("testY: ", testY)
    pre = metrics.precision_score(testY, predictions, average="weighted")
    print("Precision: {}%".format(100 * pre))
    rec = metrics.recall_score(testY, predictions, average="weighted")
    print("Recall: {}%".format(100 * rec))
    f1sc = metrics.f1_score(testY, predictions, average="weighted")
    print("f1_score: {}%".format(100 * f1sc))
    print("")
    print("Confusion Matrix:")
    conf_matrix = metrics.confusion_matrix(testY, predictions)
    print(conf_matrix)
    norm_conf_matrix = np.array(conf_matrix, dtype=np.float32) / np.sum(conf_matrix) * 100
    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(norm_conf_matrix)
    ## update summaries ###
    prediction.append(predictions)
    test.append(testY)
    accuracy.append(score[0])
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
print("Overall Precision: %.3f%% (%.3f%%)" % (mean(precision)*100.0, stdev(precision)*100.0))
print("Overall Recall: %.3f%% (%.3f%%)" % (mean(recall)*100.0, stdev(recall)*100.0))
print("Overall F1Score: %.3f%% (%.3f%%)" % (mean(f1_score)*100.0, stdev(f1_score)*100.0))
print('Confusion Matrix')
for i in range(0,10):
    print(confusion_matrix[i])
print('Normalized Confusion Matrix')
for i in range(0,10):
    print(normalised_confusion_matrix[i])