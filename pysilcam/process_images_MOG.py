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
imMOG2_arr = []
imGMG_arr = []
imMA_arr = []
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG()
subtractorMOG2 = cv.createBackgroundSubtractorMOG2()
subtractorGMG = cv.createBackgroundSubtractorGMG()

for timestamp, imraw in aqgen:
    maskMOG = subtractorMOG.apply(imraw)
    maskMOG2 = subtractorMOG2.apply(imraw)
    maskGMG = subtractorGMG.apply(imraw)

    imraw_arr.append(imraw)
    imMOG_arr.append(maskMOG)
    imMOG2_arr.append(maskMOG2)
    imGMG_arr.append(maskGMG)
    timestamp_arr.append(timestamp)

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
    imMA_arr.append(imc)

for i in range(0, 15):
    fig, ax = plt.subplots(nrows=4)
    plt.suptitle(timestamp_arr[i])
    ax[0].imshow(imraw_arr[i])
    ax[0].set_title('Original')
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].imshow(imMOG2_arr[i])
    ax[1].set_title('MOG2 ' + str(imMOG2_arr[i].shape))
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[2].imshow(imGMG_arr[i])
    ax[2].set_title('GMG ' + str(imGMG_arr[i].shape))
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])
    if i > 4:
        ax[3].imshow(imMA_arr[i-5])
        ax[3].set_title('Moving Average ' + str(imMA_arr[i-5].shape))
        ax[3].set_yticklabels([])
        ax[3].set_xticklabels([])
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")