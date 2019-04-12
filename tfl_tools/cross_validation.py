# Evaluate using Cross Validation
import os
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
from tflearn.data_utils import shuffle, image_preloader
import pysilcam.silcam_classify as sccl
from statistics import mean,stdev

# -- PATHS ---------------------------
DATABASE_PATH = '/mnt/DATA/dataset'
MODEL_PATH = '/mnt/DATA/model/modelCV'
HEADER_FILE = os.path.join(MODEL_PATH, "header.tfl.txt")         # the header file that contains the list of classes
trainset_file = os.path.join(MODEL_PATH,"imagelist_train.dat")   # the file that contains the list of images of the training dataset along with their classes
testset_file = os.path.join(MODEL_PATH,"imagelist_test.dat")     # the file that contains the list of images of the testing dataset along with their classes
set_file = os.path.join(MODEL_PATH,"imagelist.dat")     # the file that contains the list of images of the testing dataset along with their classes
IMXY = 32
SPLIT_PERCENT = 0.05   # split the train and test data i.e 0.05 is a 5% for the testing dataset and 95% for the training dataset
CHECK_POINT_FILE = "plankton-classifier.tfl.ckpt"
MODEL_FILE = "plankton-classifier.tfl"
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
print('=== Formatting database....')
classList = find_classes()
save_classes(classList)
print("CLASSLIST SIZE ", pd.read_csv(HEADER_FILE, header=None).shape[1])
# --- get file list from the folder structure
print('Import directory structure....')
fileList = import_directory_structure(classList)
# -- shuffle the dataset
print('Shuffle dataset....')
np.random.shuffle(fileList)

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = classList # ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = pd.read_csv(url, names=names)
#array = dataframe.values
# X = fileList
print('Save into a file ....')
np.savetxt(set_file, fileList, delimiter=' ', fmt='%s')
# -- call image_preloader
print('Call image_preloader ....')
X, Y = image_preloader(set_file, image_shape=(IMXY, IMXY, 3), mode='file', categorical_labels=True, normalize=True)


#X = array[:,0:8]
#Y = array[:,8]
#num_instances = len(X)

results = []
i = 0
for trainX, trainY, testX, testY in make_dataset(X,Y,10):
    i = i + 1
    # kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    # #model = LogisticRegression()
    # Build the model
    print("MODEL_PATH ", MODEL_PATH, CHECK_POINT_FILE)
    model, conv_arr, class_labels = sccl.build_model(IMXY, MODEL_PATH, CHECK_POINT_FILE)
    # Training
    print("start training round %f ...", i)
    model.fit(trainX, trainY, n_epoch=50, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=128,
              snapshot_epoch=True,
              run_id='plankton-classifier')
    # Save
    print("Saving model %f ...", i)
    MODEL_FILE = "plankton-classifier" + str(i) + ".tfl"
    model_file = os.path.join(MODEL_PATH,MODEL_FILE)
    model.save(model_file)
    # Evaluate model
    score = model.evaluate(testX, testY)
    print('Test accuracy: %0.4f%%' % (score[0] * 100))
    results.append(score[0])
    #results = model_selection.cross_val_score(model, X, Y, cv=kfold, n_jobs=-1, scoring='accuracy')
    print("Accuracy: %.4f%% (%.4f%%)" % (mean(results)*100.0, stdev(results)*100.0))

fh = open('/mnt/DATA/model/modelCV/out.txt', 'a')
fh.write("Accuracy for Round %f: %.4f%% (%.4f%%)" % i, (mean(results)*100.0, stdev(results)*100.0))
fh.close