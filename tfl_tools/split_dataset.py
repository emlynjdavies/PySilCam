import os
import h5py
import numpy as np
from make_data import MakeData
from tflearn.data_utils import build_hdf5_image_dataset



# -- PATHS ---------------------------
#DATABASE_PATH = 'Z:/DATA/dataset_test'                     # for windows running version test dataset

#DATABASE_PATH = '/mnt/DATA/silcam_classification_database'  # for ubuntu running version small dataset
DATABASE_PATH = '/mnt/DATA/dataset'                        # for ubuntu running version large dataset

#DATABASE_PATH = 'Z:/DATA/silcam_classification_database'   # for windows running version small dataset
#DATABASE_PATH = 'Z:/DATA/dataset'                          # for windows running version large dataset

set_file = os.path.join(DATABASE_PATH,"image_set.dat")      # the file that contains the list of images of the testing dataset along with their classes
WIN = ''                                                    # '_win' for windows running version
#set_file = os.path.join(DATABASE_PATH,"image_set_win.dat") # the file that contains the list of images of the testing dataset along with their classes
# -----------------------------
SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset

input_width=227
input_height=227
input_channels=3
num_classes=7
IMTEST = 'image_set_test'                           # name of the test set file
IMTRAIN = 'image_set_train'                         # name of the train set file

def split_train_test(save_split = True):
    test_filename = IMTEST + WIN + '.dat'
    train_filename = IMTRAIN + WIN + '.dat'
    test_file = os.path.join(DATABASE_PATH, test_filename)
    train_file = os.path.join(DATABASE_PATH, train_filename)
    print('Make Split')
    n_splits = 0
    data_set = MakeData(n_splits= n_splits)
    print('Split the dataset into 95% training set and 5% test set ...')
    Train, Test = data_set.split_train_test(set_file, split_percent =SPLIT_PERCENT)
    print('Test set shape ... ', Test.shape)
    print('Trainning set shape ... ', Train.shape)
    if save_split:
        np.savetxt(test_file, Test, delimiter=' ', fmt='%s')
        np.savetxt(train_file, Train, delimiter=' ', fmt='%s')
    return train_file, test_file

def split_CV(n_splits = 10, save_split = True):
    data_set = MakeData(n_splits=n_splits)
    i = 0
    for train, test in data_set.split_CV(set_file):
        i = i + 1
        round_num = str(i)
        if i < 10:
            round_num = '0' + round_num
        print('train.shape ... ', train.shape)
        print('test.shape ... ', test.shape)
        test_filename = IMTEST + round_num + WIN + '.dat'
        train_filename = IMTRAIN + round_num + WIN + '.dat'
        test_file = os.path.join(DATABASE_PATH, test_filename)
        train_file = os.path.join(DATABASE_PATH, train_filename)
        if save_split:
            print('writing to test file ', test_filename)
            np.savetxt(test_file, test, delimiter=' ', fmt='%s')
            print('writing to train file ', train_filename)
            np.savetxt(train_file, train, delimiter=' ', fmt='%s')

def build_hd5(test_file, train_file, round = ''):
    test_filename = IMTEST + str(input_width) + round + WIN + '.h5'
    print('Building hdf5 for the test set... ', test_filename)
    '''
    out_test_hd5 = os.path.join(DATABASE_PATH, test_filename)
    build_hdf5_image_dataset(test_file, image_shape=(input_width, input_height, input_channels),
                             mode='file', output_path=out_test_hd5, categorical_labels=True, normalize=True)
    test_h5f = h5py.File(out_test_hd5, 'r')
    print('Test set input shape: ', test_h5f['X'].shape)
    print('Test set output shape: ', test_h5f['Y'].shape)

    train_filename = image_set_train + str(input_width) + round + WIN + '.h5'
    print('Building hdf5 for the training set...', train_filename)
    out_train_hd5 = os.path.join(DATABASE_PATH, train_filename)
    build_hdf5_image_dataset(train_file, image_shape=(input_width, input_height, input_channels),
                             mode='file', output_path=out_train_hd5, categorical_labels=True, normalize=True)
    train_h5f = h5py.File(out_train_hd5, 'r')
    print('Training set Input shape: ', train_h5f['X'].shape)
    print('Training set Output shape: ', train_h5f['Y'].shape)
'''

'''
# for one split: training - test sets 5% and 95%
test_file = os.path.join(DATABASE_PATH, image_set_test + WIN + '.dat')
train_file = os.path.join(DATABASE_PATH, image_set_train + WIN + '.dat')
test_file, train_file = split_train_test()
build_hd5(test_file, train_file)
'''
# for cross validation splitting (default number of splits n_splits = 10)
# split_CV(n_splits = 10, save_split = True)
for i in range(1,11):
    if i < 10:
        round_num = '0' + str(i)
    else:
        round_num = str(i)
    test_filename = IMTEST + round_num + WIN + '.dat'
    print(test_filename)
    test_file = os.path.join(DATABASE_PATH, test_filename)
    train_filename = IMTRAIN + round_num + WIN + '.dat'
    print(train_filename)
    train_file = os.path.join(DATABASE_PATH, train_filename)
    if test_file and train_file:
        print('Test: ' + DATABASE_PATH, IMTEST + round_num + WIN + '.dat' +
              ' and \nTrain: ' + DATABASE_PATH, IMTRAIN + round_num + WIN + '.dat'
              + 'files exist')
        build_hd5(test_file, train_file, round)

    else:
        print('Test: ' + DATABASE_PATH, IMTEST + round_num + WIN + '.dat' +
              'or \nTrain: ' + DATABASE_PATH, IMTRAIN + round_num + WIN + '.dat'
              + 'files do not exist')
