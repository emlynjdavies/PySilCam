import os
import numpy as np
import tensorflow as tf

from tflearn.data_utils import image_preloader  # shuffle,
from statistics import mean,stdev
from make_data import MakeData
from net import Net
import h5py
import tflearn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# -- PATHS ---------------------------
#DATABASE_PATH = 'Z:/DATA/dataset_test'
#MODEL_PATH = 'Z:/DATA/model/modelCV2'
DATABASE_PATH = '/mnt/DATA/dataset'
#DATABASE_PATH = '/mnt/DATA/silcam_classification_database'
MODEL_PATH = '/mnt/DATA/model/modelAlexNet'
LOG_FILE = os.path.join(MODEL_PATH, 'AlexNetDB2_v1.out')
# -----------------------------

name='AlexNet'
input_width=224
input_height=224
input_channels=3
num_classes=7

learning_rate=0.0001  # 0.001 for OrgNet -- 0.01 for MINST -- 0.001 for CIFAR10 -- 0.001 for AlexNet
                        # 0.0001 for VGGNet -- 0.001 for GoogLeNet
momentum=0.9
keep_prob=0.5  # 0.75 for OrgNet -- 0.8 for LeNet -- 0.5 for CIFAR10 -- 0.5 for AlexNet
                # 0.5 for VGGNET -- 0.4 for GoogLeNet

n_epoch = 50  # 50
batch_size = 128 # 128
n_splits = 1  # 10 for cross_validation, 1 for one time run

i = 0
prediction = []
test = []
accuracy = []
precision = []
recall = []
f1_score = []
confusion_matrix = []
normalised_confusion_matrix = []
myNet = Net(name, input_width, input_height, input_channels, num_classes, learning_rate,
                momentum, keep_prob)
fh = open(LOG_FILE, 'w')
fh.write(name)
print(name)
'''
for i in range(0,n_splits):

    if n_splits > 1:
        i = i + 1
        round_num = str(i)
        if i < 10:
            round_num = '0' + round_num
    else:
        round_num = ''
'''


round_num = ''
out_test_hd5 = os.path.join(MODEL_PATH, 'image_set_testdb2_' + str(input_width) + round_num + ".h5")
#out_train_hd5 = os.path.join(MODEL_PATH, 'image_set_traindb2_' + str(input_width) + round_num + ".h5")
#train_h5f = h5py.File(out_train_hd5, 'r+')
test_h5f = h5py.File(out_test_hd5, 'r+')
#trainX = train_h5f['X']
#trainY = train_h5f['Y']
#print('trainX.shape ', trainX.shape, trainX[0])
#print('trainY.shape', trainY.shape, trainY[0])

testX = test_h5f['X']
testY = test_h5f['Y']
print('testX.shape ', type(testX), testX.shape, testX[0])
print('testY.shape', type(testY), testY.shape, testY[0])

tf.reset_default_graph()

tflearn.config.init_graph(seed=8888, gpu_memory_fraction=0.3, soft_placement=True) # num_cores default is All
#tflearn.config.init_graph(seed=8888, soft_placement=True) # num_cores default is All
#config = tf.ConfigProto(allow_soft_placement=True, allow_growth = True, device_count = {'GPU':2})
config = tf.ConfigProto(allow_soft_placement=True)


config.gpu_options.allocator_type='BFC'
config.gpu_options.per_process_gpu_memory_fraction=0.3
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

round_num = 'AlexNetGPU'
model_file = os.path.join(MODEL_PATH, round_num + '/plankton-classifier.tfl')

model, conv_arr = myNet.build_model(model_file)

# tf.get_variable_scope().reuse_variables()
'''
print("start training round ", round_num)
model_name = MODEL_PATH + '/' + round_num + '/plankton-classifier'
print('model_name ', model_name)
myNet.train(model, trainX, trainY, testX, testY, round_num, n_epoch, batch_size, model_name= model_name)

# Save
print("Saving model %f ..." % i)
model.save(model_file)
'''
model.load(model_file)
# Evaluate
y_pred, y_true, acc, pre, rec, f1sc, conf_matrix, norm_conf_matrix = \
    myNet.evaluate(model, testX, testY)

## update summaries ###
prediction.append(y_pred)
test.append(y_true)
accuracy.append(acc)
precision.append(pre)
recall.append(rec)
f1_score.append(f1sc)
confusion_matrix.append(conf_matrix)
normalised_confusion_matrix.append(norm_conf_matrix)


for i in range(0, n_splits):
    fh.write("\nRound ")
    if i < 10:
        j = '0' + str(i)
    fh.write(j)
    print("Round ", j)
    fh.write("\nPredictions: ")
    for el in y_pred:
        fh.write("%s " % el)
    fh.write("\ny_true: ")
    for el in y_true:
        fh.write("%s " % el)
    print("\nAccuracy: {}%".format(100*accuracy[i]))
    fh.write("\nAccuracy: {}%".format(100*accuracy[i]))
    print("Precision: {}%".format(100 * precision[i]))
    fh.write("\tPrecision: {}%".format(100 * precision[i]))
    print("Recall: {}%".format(100 * recall[i]))
    fh.write("\tRecall: {}%".format(100 * recall[i]))
    print("F1_Score: {}%".format(100 * f1_score[i]))
    fh.write("\tF1_Score: {}%".format(100 * f1_score[i]))
    print("confusion_matrix: ", confusion_matrix[i])
    print("Normalized_confusion_matrix: ", normalised_confusion_matrix[i])
fh.close
'''
print("\nOverall_Accuracy: %.3f%% " % (mean(accuracy)*100.0))
print("\nOverall_STD_Accuracy: %.3f%% " % (stdev(accuracy)*100.0))
fh.write("\nOverall_Accuracy: %.3f%% " % (mean(accuracy)*100.0))
fh.write("\nOverall_STD_Accuracy: %.3f%%" % (stdev(accuracy)*100.0))

print("\tOverall_Precision: %.3f%%" % (mean(precision)*100.0))
print("\tOverall_STD_Precision: %.3f%%" % (stdev(precision)*100.0))
fh.write("\tOverall_Precision: %.3f%% " % (mean(precision)*100.0))
fh.write("\tOverall_STD_Precision: %.3f%% " % (stdev(precision)*100.0))

print("\tOverall_Recall: %.3f%% " % (mean(recall)*100.0))
print("\tOverall_STD_Recall: %.3f%% " % (stdev(recall)*100.0))
fh.write("\tOverall_Recall: %.3f%% " % (mean(recall)*100.0))
fh.write("\tOverall_STD_Recall: %.3f%% " % (stdev(recall)*100.0))

print("\tOverall_F1Score: %.3f%% " % (mean(f1_score)*100.0))
print("\tOverall_STD_F1Score: %.3f%% " % (stdev(f1_score)*100.0))
fh.write("\tOverall_F1Score: %.3f%% " % (mean(f1_score)*100.0))
fh.write("\tOverall_STD_F1Score: %.3f%% " % (stdev(f1_score)*100.0))

print('Confusion_Matrix')
for i in range(0,n_splits):
    print(confusion_matrix[i])

print('Normalized_Confusion_Matrix')
for i in range(0,n_splits):
    print(normalised_confusion_matrix[i])
fh.close
'''

