import os
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import h5py
import tflearn
from tflearn.data_utils import build_hdf5_image_dataset

'''
Generate the data for cross validation or the normal train-test split
'''


class MakeData:
    def __init__(self, X_data = None, Y_data = None, n_splits = 0):
        '''
        Initialize the Make Data for cross validation
        :param X_data:  input data
        :param Y_data:  output data
        :param n_splits: number of splits for the cross validation (default - 10)
        '''
        self.X_data = X_data
        self.Y_data = Y_data
        self.n_splits = n_splits


    def split_train_test(self, set_file, split_percent =0.05):
        '''
        split full dataset from file into training and test sets
        :param set_file: # the file having all the list of images along with their assigned class
                split_percent #  if 0.05 the split is a 5% for the testing set and 95% for the training dataset
        '''
        file_list = pd.read_csv(set_file,sep=' ', header=None)
        Train, Test = train_test_split(file_list,
                                           test_size=split_percent,
                                           random_state=42
                                           )
        return Train, Test

    def split_CV(self, set_file):
        '''
        split full dataset from file following the cross validation approach
        :param set_file: # the file having all the list of images along with their assigned class
        '''
        file_list = pd.read_csv(set_file,sep=' ', header=None)
        seed = 7
        for train_index, test_index in \
                model_selection.KFold(n_splits=self.n_splits,shuffle=True,random_state=seed).split(file_list):
            Train, Test = file_list.iloc[train_index], file_list.iloc[test_index]
            yield Train, Test


    def makeXY(self, split_percent = 0.05):
        '''
        Split data into training and test datasets
        :param split_percent: # split the train and test data
                              #  i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(self.X_data, self.Y_data,
                                                            test_size = split_percent,
                                                            random_state = 42
                                                            ) # stratify=self.Y_data,
        print('Size of the training set ', len(X_train))
        print('Size of the output training set ', len(Y_train))
        print('Size of the test set ', len(X_test))
        print('Size of the output test set ', len(Y_test))
        print(np.unique(Y_train))
        print(np.unique(Y_test))

        return X_train, X_test, Y_train, Y_test


    def gen(self):
        '''
        Generated the cross validation data sets
        :return: X_train input training set, Y_train output training set,
                    X_test input test set, Y_test output test set
        '''
        seed = 7
        for train_index, test_index in \
                model_selection.KFold(n_splits=self.n_splits,shuffle=True,random_state=seed).split(self.X_data):
            X_train, X_test = self.X_data[train_index], self.X_data[test_index]
            Y_train, Y_test = self.Y_data[train_index], self.Y_data[test_index]


            yield X_train,Y_train,X_test,Y_test