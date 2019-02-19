from pysilcam.config import load_config, PySilcamSettings
import pysilcam.silcam_classify as sccl
import pysilcam.postprocess as scpp
import numpy as np
import pandas as pd
import skimage.io as skio
import os
from shutil import copyfile

DATABASE_PATH = 'Z:/DATA/SILCAM/silcam_classification_database'
config_file = 'Z:/DATA/SILCAM/200918/config.ini'
stats_csv_file = 'Z:/DATA/SILCAM/200918/proc/200918-STATS.csv'
filepath = 'Z:/DATA/SILCAM/200918/export'
model_path = 'Z:/DATA/model/model004/'

confidence_threshold = [0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]

DATABASE_selftaught_PATH = os.path.join(DATABASE_PATH,'../200918/','silcam_self_taught_database_0.98')

header = pd.read_csv(os.path.join(model_path, 'header.tfl.txt'))
OUTPUTS = len(header.columns)
class_labels = header.columns
print(class_labels)
print(confidence_threshold)

for cl in class_labels:
    os.makedirs(os.path.join(DATABASE_selftaught_PATH,cl),exist_ok=True)

stats = pd.read_csv(stats_csv_file)

#choice, confidence = sccl.choise_from_stats(stats)

for i,cl in enumerate(class_labels):
    class_label = 'probability_' + cl
    print(class_label)
    print(class_labels[i])

    #sstats = stats[(choice==class_label) & (confidence>confidence_threshold[i])]
    sstats = stats[(stats[class_label] > confidence_threshold[i])]
    if len(sstats)==0:
        continue

    for j in np.arange(0,len(sstats)):
        filename = sstats.iloc[j]['export name']

        im = scpp.export_name2im(filename, filepath)

        copy_to_path = os.path.join(DATABASE_selftaught_PATH,
                class_labels[i],
                filename)

        skio.imsave(copy_to_path + '.tiff', im)

        print(filename)

        #imfile = os.path.join(filepath,filename)
        #copy_to_path = os.path.join(DATABASE_selftaught_PATH,
        #        class_labels[i],
        #        filename)

        #print('from:')
        #print(imfile)

        #print('to:')
        #print(copy_to_path)

        #copyfile(imfile,copy_to_path)
