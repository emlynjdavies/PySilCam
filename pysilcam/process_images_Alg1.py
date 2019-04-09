# -*- coding: utf-8 -*-
import sys
import time
import datetime
import logging
from docopt import docopt
import numpy as np
from pysilcam import __version__
from pysilcam.acquisition import Acquire
from pysilcam.background import backgrounder, shift_and_correct, ini_background
from pysilcam.process import statextract
from pysilcam.process import image2blackwhite_accurate
import pysilcam.oilgas as scog
from pysilcam.config import PySilcamSettings
from skimage import filters
from skimage.feature import hog
from skimage import exposure
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.color import rgb2gray
import os
import pysilcam.silcam_classify as sccl
import multiprocessing
from multiprocessing.managers import BaseManager
from queue import LifoQueue
import psutil
from shutil import copyfile
import warnings
import pandas as pd

#from pylab import *
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.filters import threshold_otsu, threshold_adaptive, try_all_threshold
import random as rng

# Z:/DATA/SILCAM/250718/config.ini Z:/DATA/SILCAM/RAW/250718 --nomultiproc
#config_filename = "Z:/DATA/SILCAM/Working/config.ini"
config_filename = "/mnt/DATA/SILCAM/Working/config.ini"
#datapath = "Z:/DATA/SILCAM/Working/RAW250718"
datapath = "/mnt/DATA/SILCAM/Working/RAW250718"
discWrite = False
###################################################

####################################################
# Loading config file
# Load the configuration, create settings object
settings = PySilcamSettings(config_filename)
# Print configuration to screen
print('---- CONFIGURATION ----\n')
settings.config.write(sys.stdout)
print('-----------------------\n')

# Configure logging
logging.basicConfig(level=getattr(logging, settings.General.loglevel)) # configure_logger(settings.General)
logger = logging.getLogger(__name__ + '.silcam_process')

logger.info('Processing path: ' + datapath)

###################################################
rng.seed(12345)

imraw_arr = []
imMOG_arr = []
imMOGSeg_arr = []
imMOGSegO_arr = []
thresh_arr = []
dist_arr =[]
sure_fg_arr = []
markers_arr = []
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG2(history=5, varThreshold=25, detectShadows=False)

for timestamp, imraw in aqgen:
    maskMOG = subtractorMOG.apply(imraw)
    imraw_arr.append(np.copy(imraw))
    imMOG_arr.append(np.copy(maskMOG))
    timestamp_arr.append(timestamp)
    # Finding foreground area
    # opening operation
    ret, thresh = cv.threshold(maskMOG, 0, 255,
                                 cv.THRESH_BINARY_INV +
                                 cv.THRESH_OTSU)
    thresh_arr.append(np.copy(thresh))
    ###
    # Noise removal using Morphological
    # opening operation
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE,
                               kernel, iterations=2)
    # Background area using Dialation
    bg = cv.dilate(opening, kernel, iterations=3)
    # Finding foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    dist_arr.append(np.copy(dist_transform))
    ret, fg = cv.threshold(dist_transform, 0.7
                             * dist_transform.max(), 255, 0)
    sure_fg_arr.append(np.copy(fg))
    ###
    ####  WATERSHED ALGORITHM #################################
    # Marker labelling
    fg = np.uint8(fg)
    unknown = cv.subtract(bg,fg)
    ret, markers = cv.connectedComponents(fg)
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers_arr.append(markers)
    markersO = markers

    new_gray = cv.cvtColor(maskMOG, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(new_gray, markers)
    maskMOG[markers == -1] = 255
    imMOGSeg_arr.append(maskMOG)

    markersO = cv.watershed(new_gray, markersO)
    imraw[markersO == -1] = [255,0,0]
    imMOGSegO_arr.append(imraw)





for i in range(0, 15):
    fig, ax = plt.subplots(nrows=4, ncols=2)
    plt.suptitle(timestamp_arr[i])
    ax[0, 0].imshow(imraw_arr[i])
    ax[0, 0].set_title('Original')
    ax[0, 1].imshow(imMOG_arr[i])
    ax[0, 1].set_title('MOG' + str(imMOG_arr[i].shape))

    ax[1, 0].imshow(thresh_arr[i])
    ax[1, 0].set_title('thresh ' + str(thresh_arr[i].shape))
    ax[1, 1].imshow(dist_arr[i])
    ax[1, 1].set_title('dist' + str(dist_arr[i].shape))

    ax[2, 0].imshow(sure_fg_arr[i])
    ax[2, 0].set_title('sure_fg ' + str(sure_fg_arr[i].shape))
    ax[2, 1].imshow(markers_arr[i])
    ax[2, 1].set_title('markers ' + str(markers_arr[i].shape))

    ax[3, 0].imshow(imMOGSeg_arr[i])
    ax[3, 0].set_title('imMOGSeg ' + str(imMOGSeg_arr[i].shape))
    ax[3, 1].imshow(imMOGSegO_arr[i])
    ax[3, 1].set_title('imMOGSegOrig ' + str(imMOGSegO_arr[i].shape))

    for j in range(0, 4):
        for k in range(0, 2):
            ax[j, k].set_yticklabels([])
            ax[j, k].set_xticklabels([])

    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")

'''imraw_arr = []
imMOG_arr = []

thresh_arr = []
dist_arr =[]
sure_fg_arr = []
markers_arr = []
imMOGSeg_arr = []
imMOGSegO_arr = []
'''
