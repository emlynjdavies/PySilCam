import numpy as np
import os
import pandas as pd

# -----------------------------
DATABASE_PATH = '/mnt/DATA/dataset_balanced'

HEADER_FILE = os.path.join(DATABASE_PATH, "header.tfl.txt")         # the header file that contains the list of classes
data_file = os.path.join(DATABASE_PATH,'image_set.dat')

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
print('Save into file ....')
np.savetxt(data_file, fileList, delimiter=' ', fmt='%s')

