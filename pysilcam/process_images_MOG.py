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
imMA_arr = []
imMOGSeg_arr = []
imMASeg_arr = []
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG2(history=5, varThreshold=25, detectShadows=False)

for timestamp, imraw in aqgen:
    maskMOG = subtractorMOG.apply(imraw)
    imraw_arr.append(imraw)
    maskMOG_cp = np.copy(maskMOG)
    imMOG_arr.append(maskMOG_cp)
    timestamp_arr.append(timestamp)
    # Finding foreground area
    # closing operation
    kernel = np.ones((3, 3), np.uint8)
    ret, thresh = cv.threshold(maskMOG, 0, 255,
                                 cv.THRESH_BINARY_INV +
                                 cv.THRESH_OTSU)
    ###
    # Noise removal using Morphological
    # closing operation
    # kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE,
                               kernel, iterations=2)
    # Background area using Dialation
    bg = cv.dilate(closing, kernel, iterations=1)
    # Finding foreground area
    dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 0)
    ret, fg = cv.threshold(dist_transform, 0.02
                             * dist_transform.max(), 255, 0)
    ###
    ####  WATERSHED ALGORITHM #################################
    # Marker labelling
    fg = np.uint8(fg)
    ret, markers = cv.connectedComponents(fg)
    markers = markers + 1
    new_gray = cv.cvtColor(maskMOG, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(new_gray, markers)
    maskMOG[markers == -1] = [255]
    imMOGSeg_arr.append(maskMOG)

###################################################
aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)

#Get number of images to use for background correction from config
print('* Initializing background image handler')
bggen = backgrounder(5, aqgen,
                     bad_lighting_limit = None,
                     real_time_stats=False)

for i, (timestamp, imc, imraw) in enumerate(bggen):
    imc_cp = np.copy(imc)
    imMA_arr.append(imc_cp)

    # Finding foreground area
    # closing operation
    kernel = np.ones((3, 3), np.uint8)
    new_imc = cv.cvtColor(imraw_arr[i], cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(new_imc, 0, 255,
                               cv.THRESH_BINARY_INV +
                               cv.THRESH_OTSU)
    ###
    # Noise removal using Morphological
    # closing operation
    # kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE,
                              kernel, iterations=2)
    # Background area using Dialation
    bg = cv.dilate(closing, kernel, iterations=1)
    # Finding foreground area
    dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 0)
    ret, fg = cv.threshold(dist_transform, 0.02
                           * dist_transform.max(), 255, 0)
    ###
    ####  WATERSHED ALGORITHM #################################
    # Marker labelling
    fg = np.uint8(fg)
    ret, markers = cv.connectedComponents(fg)
    markers = markers + 1
    #new_gray = cv.cvtColor(imc, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(imc, markers)
    imc[markers == -1] = [255]
    imMASeg_arr.append(imc)

for i in range(0, 15):
    fig, ax = plt.subplots(nrows=3, ncols=2)
    plt.suptitle(timestamp_arr[i])
    ax[0, 0].imshow(imraw_arr[i])
    ax[0, 0].set_title('Original')
    ax[1, 0].imshow(imMOG_arr[i])
    ax[1, 0].set_title('MOG hist=5 thres=25 ' + str(imMOG_arr[i].shape))
    ax[1, 1].imshow(imMOGSeg_arr[i])
    ax[1, 1].set_title('MOG Seg ' + str(imMOGSeg_arr[i].shape))
    if i > 4:
        ax[2, 0].imshow(imMA_arr[i-5])
        ax[2, 0].set_title('Moving Average ' + str(imMA_arr[i-5].shape))
        ax[2, 1].imshow(imMASeg_arr[i - 5])
        ax[2, 1].set_title('Moving Average Seg ' + str(imMASeg_arr[i - 5].shape))

    for j in range(0, 3):
        for k in range(0,2):
            ax[j, k].set_yticklabels([])
            ax[j, k].set_xticklabels([])

    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")