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
win = ''                                                    # '_win' for windows running version
#set_file = os.path.join(DATABASE_PATH,"image_set_win.dat") # the file that contains the list of images of the testing dataset along with their classes
# -----------------------------
SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset

input_width=227
input_height=227
input_channels=3
num_classes=7


#n_splits = 0
#data_set = MakeData(n_splits= n_splits)
#print('Split the dataset into 95% training set and 5% test set ...')
#Train, Test = data_set.split_train_test(set_file, split_percent =SPLIT_PERCENT)

print('Building hdf5 for the test set...')
#print('Test set shape ... ', Test.shape)
test_file = os.path.join(DATABASE_PATH, 'image_set_test' + win + '.dat')
#np.savetxt(test_file, Test, delimiter=' ', fmt='%s')
out_test_hd5 = os.path.join(DATABASE_PATH, 'image_set_test' + str(input_width) + win + ".h5")
build_hdf5_image_dataset(test_file, image_shape=(input_width, input_height, input_channels),
                         mode='file', output_path=out_test_hd5, categorical_labels=True, normalize=True)
test_h5f = h5py.File(out_test_hd5, 'r')
print('Test set input shape: ', test_h5f['X'].shape)
print('Test set output shape: ', test_h5f['Y'].shape)

print('Building hdf5 for the training set...')
#print('Trainning set shape ... ', Train.shape)
train_file = os.path.join(DATABASE_PATH, 'image_set_train' + win + '.dat')
#np.savetxt(train_file, Train, delimiter=' ', fmt='%s')
out_train_hd5 = os.path.join(DATABASE_PATH, 'image_set_train' + str(input_width) + win + ".h5")
build_hdf5_image_dataset(train_file, image_shape=(input_width, input_height, input_channels),
                         mode='file', output_path=out_train_hd5, categorical_labels=True, normalize=True)
train_h5f = h5py.File(out_train_hd5, 'r')

print('Training set Input shape: ', train_h5f['X'].shape)
print('Training set Output shape: ', train_h5f['Y'].shape)

