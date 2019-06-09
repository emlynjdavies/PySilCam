import os
import numpy as np
import tensorflow as tf

from tflearn.data_utils import image_preloader  # shuffle,
from statistics import mean,stdev
from make_data import MakeData
from net import Net
import h5py
import tflearn

from train_over_gpus import train_multi_gpu as mg
from datetime import datetime
import os.path
import re
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# -- PATHS ---------------------------
#DATABASE_PATH = 'Z:/DATA/dataset_test'
#MODEL_PATH = 'Z:/DATA/model/modelCV2'
#DATABASE_PATH = '/mnt/DATA/dataset'
#DATABASE_PATH = '/mnt/DATA/silcam_classification_database'
MODEL_PATH = '/mnt/DATA/model/modelAlexNet'
LOG_FILE = os.path.join(MODEL_PATH, 'AlexNetDB1.out')
# -----------------------------
name='AlexNet'
input_width=224
input_height=224
input_channels=3
num_classes=7

learning_rate=0.001  # 0.001 for OrgNet -- 0.01 for MINST -- 0.001 for CIFAR10 -- 0.001 for AlexNet
                        # 0.0001 for VGGNet -- 0.001 for GoogLeNet
momentum=0.9
keep_prob=0.4  # 0.75 for OrgNet -- 0.8 for LeNet -- 0.5 for CIFAR10 -- 0.5 for AlexNet
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
VGGNet = Net(name, input_width, input_height, input_channels, num_classes, learning_rate,
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
round_path = 'AlexNetGPUSMALL'
model_file = os.path.join(MODEL_PATH, round_path + '/plankton-classifier.tfl')
round_num = ''
out_test_hd5 = os.path.join(MODEL_PATH, 'image_set_test' + str(input_width) + round_num + ".h5")
out_train_hd5 = os.path.join(MODEL_PATH, 'image_set_train' + str(input_width) + round_num + ".h5")
train_h5f = h5py.File(out_train_hd5, 'r+')
test_h5f = h5py.File(out_test_hd5, 'r+')
trainX = train_h5f['X']
trainY = train_h5f['Y']
testX = test_h5f['X']
testY = test_h5f['Y']
print('testX.shape ', type(testX), testX.shape, testX[0])
print('testY.shape', type(testY), testY.shape, testY[0])

print(mg.num_gpus)
print(mg.TOWER_NAME)

# ###########################################################################
"""Train CIFAR-10 for a number of steps."""
with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (trainX.shape[0] /
                             batch_size / mg.num_gpus)
    decay_steps = int(num_batches_per_epoch * mg.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(learning_rate,
                                    global_step,
                                    decay_steps,
                                    mg.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    # Get images and labels for CIFAR-10.
    images, labels = tf.convert_to_tensor(trainX, dtype=tf.float32), \
                     tf.convert_to_tensor(trainY, dtype=tf.int32) #np.amax(trainY, axis=1) #trainY[trainY.argmax(axis=0)]
    print('images ', images.shape, images[0], type(images))
    print('labels', labels.shape, labels[0], type(labels))
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=2 * mg.num_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(mg.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (mg.TOWER_NAME, i)) as scope:
                    # Dequeues one batch for the GPU
                    image_batch, label_batch = batch_queue.dequeue()
                    '''
                    ################################################################
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    model, conv_arr = VGGNet.build_model(model_file)
                    loss = mg.tower_loss(scope, model, label_batch)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = mg.average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        mg.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=mg.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(MODEL_PATH + '/' + round_path, sess.graph)

    for step in range(mg.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = batch_size * mg.num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / mg.num_gpus

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == mg.max_steps:
            checkpoint_path = os.path.join(MODEL_PATH + '/' + round_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
#######################################################################################
'''
# #######################################################################################            
'''
tf.reset_default_graph()
tflearn.config.init_graph(seed=8888, gpu_memory_fraction=0.9, soft_placement=True) # num_cores default is All
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type='BFC'
config.gpu_options.per_process_gpu_memory_fraction=0.9
sess = tf.Session(config=config)
round_num = 'GoogleNetGPUSMALL'
model_file = os.path.join(MODEL_PATH, round_num + '/plankton-classifier.tfl')

# with tf.device('/gpu:0'):
for d in ['/device:GPU:0', '/device:GPU:1']:
    with tf.device(d):
        model, conv_arr = VGGNet.build_model(model_file)

with tf.device('/cpu:0'):
    print("start training round ", round_num)
    VGGNet.train(model, trainX, trainY, testX, testY, round_num, n_epoch, batch_size)
    # Save
    print("Saving model %f ..." % i)
    model.save(model_file)
    # Evaluate
    y_pred, y_true, acc, pre, rec, f1sc, conf_matrix, norm_conf_matrix = \
        VGGNet.evaluate(model, testX, testY)

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

