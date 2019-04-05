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
imraw_arr = []
imMOG_arr = []
imSegmented_arr = []
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG2()

for timestamp, imraw in aqgen:
    gray = cv.cvtColor(imraw, cv.COLOR_RGB2GRAY)
    maskMOG = subtractorMOG.apply(imraw)

    # Finding foreground area
    # closing operation
    kernel = np.ones((3, 3), np.uint8)
    ret2, thresh2 = cv.threshold(maskMOG, 0, 255,
                                 cv.THRESH_BINARY_INV +
                                 cv.THRESH_OTSU)
    ###
    # Noise removal using Morphological
    # closing operation
    #kernel = np.ones((3, 3), np.uint8)
    closing2 = cv.morphologyEx(thresh2, cv.MORPH_CLOSE,
                              kernel, iterations=2)

    # Background area using Dialation
    bg2 = cv.dilate(closing2, kernel, iterations=1)

    # Finding foreground area
    dist_transform2 = cv.distanceTransform(closing2, cv.DIST_L2, 0)
    ret2, fg2 = cv.threshold(dist_transform2, 0.02
                           * dist_transform2.max(), 255, 0)
    ###

    ####  WATERSHED ALGORITHM #################################
    # Marker labelling
    fg2 = np.uint8(fg2)
    ret, markers = cv.connectedComponents(fg2)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    markers = cv.watershed(imraw, markers)
    imcp = np.copy(imraw)
    imraw[markers == -1] = [255, 0, 0]
    #####################################


    x, y, z = imraw.shape
    print("timestamp ", timestamp)
    print("Image raw shape imraw.shape( ", x,y,z)
    gray_frame = cv.cvtColor(imraw, cv.COLOR_RGB2GRAY)
    imraw_arr.append(imcp)
    imMOG_arr.append(maskMOG)
    imSegmented_arr.append(imraw)
    timestamp_arr.append(timestamp)


for i in range(0, 11):
    fig, ax = plt.subplots(nrows=3)
    plt.suptitle(timestamp_arr[i])
    ax[0].imshow(imraw_arr[i])
    ax[0].set_title('Original')
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].imshow(imSegmented_arr[i])
    ax[1].set_title('Segmented')
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[2].imshow(imMOG_arr[i])
    ax[2].set_title('MOG')
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")