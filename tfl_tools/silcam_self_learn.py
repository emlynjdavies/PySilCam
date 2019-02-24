from pysilcam.config import load_config, PySilcamSettings
import pysilcam.silcam_classify as sccl
import pysilcam.postprocess as scpp
import numpy as np
import pandas as pd
import skimage.io as skio
import os
from shutil import copyfile


# ======================================================
def extract_middle(stats):
    '''
    Temporary cropping solution due to small window in AUV
    '''
    print('initial stats length:', len(stats))
    r = np.array(((stats['maxr'] - stats['minr']) / 2) + stats['minr'])
    c = np.array(((stats['maxc'] - stats['minc']) / 2) + stats['minc'])

    points = []
    for i in range(len(c)):
        points.append([(r[i], c[i])])

    pts = np.array(points)
    pts = pts.squeeze()

    # plt.plot(pts[:, 0], pts[:, 1], 'k.')
    # plt.axis('equal')

    ll = np.array([500, 500])  # lower-left
    ur = np.array([1750, 1750])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    inbox = pts[inidx]
    print('inbox shape:', inbox.shape)

    stats = stats[inidx]

    # plt.plot(inbox[:,0], inbox[:,1], 'r.')
    # plt.axis('equal')

    print('len stats', len(stats))
    return stats
# ======================================================

DATABASE_PATH = 'Z:/DATA/SILCAM/silcam_classification_database_model004'
config_file = 'Z:/DATA/SILCAM/250718/config.ini'
stats_csv_file = 'Z:/DATA/SILCAM/250718/proc/250718-STATS.csv'
filepath = 'Z:/DATA/SILCAM/250718/export'
model_path = 'Z:/DATA/model/model004/'

confidence_threshold = [0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]

DATABASE_selftaught_PATH = os.path.join(DATABASE_PATH,'../250718/','silcam_self_taught_database_border_clean_0.98')

header = pd.read_csv(os.path.join(model_path, 'header.tfl.txt'))
OUTPUTS = len(header.columns)
class_labels = header.columns
print(class_labels)
print(confidence_threshold)

for cl in class_labels:
    os.makedirs(os.path.join(DATABASE_selftaught_PATH,cl),exist_ok=True)

stats0 = pd.read_csv(stats_csv_file)
stats = extract_middle(stats0)
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
