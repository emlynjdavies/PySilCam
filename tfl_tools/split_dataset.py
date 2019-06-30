import os
import h5py
import numpy as np
from make_data import MakeData
from tflearn.data_utils import build_hdf5_image_dataset



# -- PATHS ---------------------------
#DATABASE_PATH = 'Z:/DATA/dataset_test'                     # for windows running version test dataset

#DATABASE_PATH = '/mnt/DATA/silcam_classification_database'  # for ubuntu running version small dataset
# DATABASE_PATH = '/mnt/DATA/dataset'                        # for ubuntu running version large dataset

#DATABASE_PATH = 'Z:/DATA/silcam_classification_database'   # for windows running version small dataset
#DATABASE_PATH = 'Z:/DATA/dataset'                          # for windows running version large dataset

#set_file = os.path.join(DATABASE_PATH,"image_set.dat")      # the file that contains the list of images of the testing dataset along with their classes
WIN = ''                                                    # '_win' for windows running version
#set_file = os.path.join(DATABASE_PATH,"image_set_win.dat") # the file that contains the list of images of the testing dataset along with their classes
# -----------------------------
SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset

#input_width=32
#input_height=32
#input_channels=3
#num_classes=7
IMSETTEST = 'image_set_test'                           # name of the test set file
IMSETTRAIN = 'image_set_train'                         # name of the train set file

def split_train_test(db_path, set_file, save_split = True):
    test_filename = IMSETTEST + WIN + '.dat'
    train_filename = IMSETTRAIN + WIN + '.dat'
    test_file = os.path.join(db_path, test_filename)
    train_file = os.path.join(db_path, train_filename)
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

def split_CV(db_path, set_file, n_splits = 10, save_split = True):
    data_set = MakeData(n_splits=n_splits)
    i = 0
    for train, test in data_set.split_CV(set_file):
        i = i + 1
        round_num = str(i)
        if i < 10:
            round_num = '0' + round_num
        print('train.shape ... ', train.shape)
        print('test.shape ... ', test.shape)
        test_filename = IMSETTEST + round_num + WIN + '.dat'
        train_filename = IMSETTRAIN + round_num + WIN + '.dat'
        test_file = os.path.join(db_path, test_filename)
        train_file = os.path.join(db_path, train_filename)
        if save_split:
            print('writing to test file ', test_filename)
            np.savetxt(test_file, test, delimiter=' ', fmt='%s')
            print('writing to train file ', train_filename)
            np.savetxt(train_file, train, delimiter=' ', fmt='%s')

def build_hd5(db_path, test_file, train_file, input_width, input_height, input_channels = 3, round = ''):
    test_filename = IMSETTEST + str(input_width) + round + WIN + '.h5'
    print('Building hdf5 for the test set... ', test_filename)
    out_test_hd5 = os.path.join(db_path, test_filename)
    build_hdf5_image_dataset(test_file, image_shape=(input_width, input_height, input_channels),
                             mode='file', output_path=out_test_hd5, categorical_labels=True, normalize=True)
    test_h5f = h5py.File(out_test_hd5, 'r')
    print('Test set input shape: ', test_h5f['X'].shape)
    print('Test set output shape: ', test_h5f['Y'].shape)

    train_filename = IMSETTRAIN + str(input_width) + round + WIN + '.h5'
    print('Building hdf5 for the training set... ', train_filename)
    out_train_hd5 = os.path.join(db_path, train_filename)
    build_hdf5_image_dataset(train_file, image_shape=(input_width, input_height, input_channels),
                             mode='file', output_path=out_train_hd5, categorical_labels=True, normalize=True)
    train_h5f = h5py.File(out_train_hd5, 'r')
    print('Training set Input shape: ', train_h5f['X'].shape)
    print('Training set Output shape: ', train_h5f['Y'].shape)


####################
'''
print('building h5 for DBII ...')
# for one split: training - test sets 5% and 95%
test_file = os.path.join(DATABASE_PATH, IMSETTEST + '_db2' + WIN + '.dat')
train_file = os.path.join(DATABASE_PATH, IMSETTEST + '_db2' + WIN + '.dat')
#train_file, test_file = split_train_test()
print('building 32x32 dataset ...')
build_hd5(test_file, train_file, input_width = 32, input_height = 32, round='_db2')
print('done')
print('building 64x64 dataset ...')
build_hd5(test_file, train_file, input_width = 64, input_height = 64, round='_db2')
print('done')
print('building 128x128 dataset ...')
build_hd5(test_file, train_file, input_width = 128, input_height = 128, round='_db2')
print('done')
'''

'''
print('building h5 for DBI ...')
DATABASE_PATH = '/mnt/DATA/silcam_classification_database'

set_file = os.path.join(DATABASE_PATH,"image_set.dat") # the file that contains the list of images of the testing dataset along with their classes

# for one split: training - test sets 5% and 95%
test_file = os.path.join(DATABASE_PATH, IMSETTEST + '_db1' + WIN + '.dat')
train_file = os.path.join(DATABASE_PATH, IMSETTEST + '_db1' + WIN + '.dat')
train_file, test_file = split_train_test(db_path=DATABASE_PATH, set_file=set_file)
print('building 32x32 dataset ...')
# def build_hd5(db_path, test_file, train_file, input_width, input_height, input_channels = 3, round = ''):
build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
          input_width = 32, input_height = 32, round='_db1')

print('done')
print('building 224x224 dataset ...')
build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
          input_width = 224, input_height = 224, round='_db1')
print('done')
print('building 227x227 dataset ...')
build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
          input_width = 227, input_height = 227, round='_db1')
print('done')


print('building 64x64 dataset ...')
build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
          input_width = 64, input_height = 64, round='_db1')
print('done')
print('building 128x128 dataset ...')
build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
          input_width = 128, input_height = 128, round='_db1')
print('done')
'''

print('building h5 for DBI ...')
DATABASE_PATH = '/mnt/DATA/silcam_classification_database'
# for cross validation splitting (default number of splits n_splits = 10)
#split_CV(db_path=DATABASE_PATH, set_file=set_file, n_splits = 10, save_split = True)

for i in range(1,11):
    if i < 10:
        round_num = '0' + str(i)
    else:
        round_num = str(i)
    test_filename = IMSETTEST + round_num + WIN + '.dat'
    print(test_filename)
    test_file = os.path.join(DATABASE_PATH, test_filename)
    train_filename = IMSETTRAIN + round_num + WIN + '.dat'
    print(train_filename)
    train_file = os.path.join(DATABASE_PATH, train_filename)
    if test_file and train_file:
        print('Test: ' + DATABASE_PATH, IMSETTEST + round_num + WIN + '.dat' +
              ' and \nTrain: ' + DATABASE_PATH, IMSETTRAIN + round_num + WIN + '.dat'
              + 'files exist')
        build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
                  input_width=32, input_height=32, round=round_num)
        build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
                  input_width=64, input_height=64, round=round_num)
        build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
                  input_width=128, input_height=128, round=round_num)
        build_hd5(db_path=DATABASE_PATH, test_file=test_file, train_file=train_file,
                  input_width=224, input_height=224, round=round_num)

        #build_hd5(test_file, train_file, round=round_num)

    else:
        print('Test: ' + DATABASE_PATH, IMSETTEST + round_num + WIN + '.dat' +
              'or \nTrain: ' + DATABASE_PATH, IMSETTRAIN + round_num + WIN + '.dat'
              + 'files do not exist')
