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
imc_arr = []
imMOG_arr = []
imMOG2_arr = []
imMOGL_arr = []
imKNN_arr = []
imDiff_arr = []
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG2()
subtractorMOG2 = cv.createBackgroundSubtractorMOG2(history=3, varThreshold=25, detectShadows=False)
subtractorMOGL = cv.createBackgroundSubtractorMOG2(history=3, detectShadows=False)
subtractorKNN = cv.createBackgroundSubtractorKNN(history=3, dist2Threshold=25, detectShadows=False)
count = 0
first_frame = None

for timestamp, imraw in aqgen:
    if count == 0:
        count = count + 1
        first_frame = imraw
        first_gray = cv.cvtColor(first_frame, cv.COLOR_RGB2GRAY)
    maskMOG = subtractorMOG.apply(imraw)
    maskMOG2 = subtractorMOG2.apply(imraw)
    maskKNN = subtractorKNN.apply(imraw)
    maskMOGL = subtractorMOGL.apply(imraw, learningRate=0.1)

    x, y, z = imraw.shape
    print("timestamp ", timestamp)
    print("Image raw shape imraw.shape( ", x,y,z)
    gray_frame = cv.cvtColor(imraw, cv.COLOR_RGB2GRAY)
    difference = cv.absdiff(first_gray, gray_frame)
    _, difference = cv.threshold(difference, 25, 255, cv.THRESH_BINARY)
    imraw_arr.append(imraw)
    imMOG_arr.append(maskMOG)
    imMOG2_arr.append(maskMOG2)
    imMOGL_arr.append(maskMOGL)
    imKNN_arr.append(maskKNN)
    imDiff_arr.append(difference)
    timestamp_arr.append(timestamp)

aq2=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
#Get number of images to use for background correction from config
print('* Initializing background image handler')
aqgen2=aq2.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
bggen = backgrounder(3, aqgen2,
                     bad_lighting_limit = None,
                     real_time_stats=False)

for i, (timestamp, imc, imraw) in enumerate(bggen):
    imc_arr.append(imc)

for i in range(0, 11):
    t_arr = ['Original Image', 'Moving Avg', 'MOG Default', 'MOG2 Threshold 25', 'Mask KNN', 'MOG learning rate 0.1']
    fig, ax = plt.subplots(nrows=2,ncols=3)
    plt.suptitle(timestamp_arr[i])
    ax[0, 0].imshow(imraw_arr[i])
    ax[0, 0].set_title(t_arr[0])
    ax[0, 1].imshow(imMOG_arr[i])
    ax[0, 1].set_title(t_arr[2])
    ax[1, 1].imshow(imMOG2_arr[i])
    ax[1, 1].set_title(t_arr[3])
    ax[1, 0].imshow(imKNN_arr[i])
    ax[1, 0].set_title(t_arr[4])
    ax[1, 2].imshow(imMOGL_arr[i])
    ax[1, 2].set_title(t_arr[5])
    if i > 2:
        ax[0, 2].imshow(imc_arr[i-3])
        ax[0, 2].set_title(t_arr[1])
    for j in range(0,2):
        for k in range(0,3):
            ax[j, k].set_yticklabels([])
            ax[j, k].set_xticklabels([])
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")