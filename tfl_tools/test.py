""" Linear Regression Example """
from __future__ import absolute_import, division, print_function
from sklearn import model_selection, metrics
import tflearn
import tensorflow as tf
import numpy as np
from statistics import mean,stdev
import tflearn.helpers.summarizer as s

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s
def make_dataset(X_data, Y_data, n_splits):
    seed = 7
    for train_index, test_index in model_selection.KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(X_data):
        print('train_index test_index ' , train_index, test_index)
        #print('X= ', X[train_index])
        #print('Y= ', Y[train_index])
        #X_train, X_test = X_data[train_index], X_data[test_index]
        #y_train, y_test = y_data[train_index], y_data[test_index]
        X_train = []
        Y_train = []
        for train_el in train_index:
            #print('train_el', train_el, X_data[train_el], Y_data[train_el])
            X_train.append(X_data[train_el])
            Y_train.append(Y_data[train_el])
        X_test = []
        Y_test = []
        for test_el in test_index:
            # print('test_el', test_el, X_data[test_el], Y_data[test_el])
            X_test.append(X_data[test_el])
            Y_test.append(Y_data[test_el])
        yield np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)

def precision_score(prediction, target, inputs=None):
    return metrics.average_precision_score(target, prediction) # , average='weighted'
# tf.metrics.precision(testY, predict, name="precision")
def recall_score(prediction, target, inputs=None):
    return metrics.recall_score(target, prediction, average='weighted')
def f1_score(prediction, target, inputs=None):
        return metrics.f1_score(target, prediction, average='weighted')

class F1Score():
    def __init__(self, name="F1Score"):
        super(F1Score, self).__init__(name)
        self.tensor = None

    def build(self, predictions, targets, inputs=None):
        with tf.name_scope('F1Score'): # <--------- name scope
            precision, _pop = tf.metrics.precision(targets, predictions, name="precision")
            recall, _rop = tf.metrics.recall(targets, predictions, name="recall")
            self.tensor = 2 * (precision * recall) / (precision + recall)
        self.built = True
        self.tensor.m_name = self.name
        return self.tensor

    def get_tensor(self):
        return self.tensor
    #return tf.reduce_sum(tf.abs(prediction - target), name='l1')
# Regression data
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
#Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]
Y = [1,2,2,3,1,1,3,2,2,1,2,3,1,2,2,2,1]

'''

print("\nRegression result:")
print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(m.predict([3.2, 3.3, 3.4]))
# should output (close, not exact) y = [1.5315033197402954, 1.5585315227508545, 1.5855598449707031]
'''
i = 0
results = []
for trainX, trainY, testX, testY in make_dataset(X, Y, 10):
    i = i + 1
    tf.reset_default_graph()
    round_num = str(i)
    if i < 10:
        round_num = '0' + round_num
    #model = LogisticRegression()
    # Linear Regression graph
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
    results.append(score[0])
    #results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))
    #print("\nTest prediction for x = 3.2, 3.3, 3.4:")
    #print(model.predict([3.2, 3.3, 3.4]))
    # prediction, target, inputs
    print("\nTest prediction for x = ",testX)
    #predictions = model.predict(testX)
    predictions = [int(i) for i in model.predict(testX)]
    print("predictions: " , predictions)
    print("testY: ", testY)
    print("Precision: {}%".format(100 * metrics.precision_score(testY, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(testY, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(testY, predictions, average="weighted")))
    #print("Precision: {}%".format(100 * precision_score(predict,testY)))

    #print("F1Score: {}",tflearn.metrics.Metric.build(F1Score,predict,testY, inputs=None))

    # print("precision", pre , pre_op)
    # print("Precision:{}%".format(100 * pre))
    #print("Precision: {}%".format(100 * precision_score(predict, testY)))
    '''
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    pre, pre_op = tf.metrics.precision(testY, predict)
    p = sess.run(pre_op)
    print("precision %f", p)
    
    #####
    x = tf.placeholder(tf.float32, )
    y = tf.placeholder(tf.float32, )
    pre, pre_op = tf.metrics.precision(testY,predict)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    p = sess.run(pre_op, feed_dict={x: predict, y: testY})  # precision
    print("precision %f", p)'''


    #s.summarize(metrics.precision_score(testY, model.predict(testX), average="weighted"))  # Summarize anything.

print("Accuracy: %.3f%% (%.3f%%)" % (mean(results)*100.0, stdev(results)*100.0))


#print("Precision: {}%".format(100*precision_score(testY, [3.2, 3.3, 3.4], testX)))
#print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
#print("Recall: {}%".format(100*recall_score(testY, [3.2, 3.3, 3.4], testX)))
#print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
#print("f1_score: {}%".format(100*f1_score(testY, [3.2, 3.3, 3.4], testX)))
#print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

