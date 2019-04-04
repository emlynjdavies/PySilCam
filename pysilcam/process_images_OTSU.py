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
imGRGB_arr = []
imGG_arr = []
imMOG_arr = []
imMOGRGB_arr = []
imMOGG_arr = []
imOTSU_arr =[]
imOTSUTH_arr =[]
imOTSURGB_arr =[]
imOTSURGBTH_arr =[]
imOTSUG_arr =[]
imOTSUGTH_arr =[]
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG2()

for timestamp, imraw in aqgen:
    gray = cv.cvtColor(imraw, cv.COLOR_RGB2GRAY)
    maskMOG = subtractorMOG.apply(imraw)
    ret, thresh = cv.threshold(gray, 0, 255,
                                cv.THRESH_BINARY_INV +
                                cv.THRESH_OTSU)

    blur = cv.GaussianBlur(imraw, (5, 5), 0)
    gray_blur = cv.cvtColor(imraw, cv.COLOR_RGB2GRAY)
    ret2, thresh2 = cv.threshold(gray_blur, 0, 255,
                                 cv.THRESH_BINARY_INV +
                                 cv.THRESH_OTSU)
    blur2 = cv.GaussianBlur(gray, (5, 5), 0)
    ret3, thresh3 = cv.threshold(blur2, 0, 255,
                                 cv.THRESH_BINARY_INV +
                                 cv.THRESH_OTSU)
    maskMOGRGB = subtractorMOG.apply(blur)
    maskMOGG = subtractorMOG.apply(blur2)

    x, y, z = imraw.shape
    print("timestamp ", timestamp)
    print("Image raw shape imraw.shape( ", x,y,z)
    gray_frame = cv.cvtColor(imraw, cv.COLOR_RGB2GRAY)
    imraw_arr.append(imraw)
    imGRGB_arr.append(blur)
    imGG_arr.append(blur2)
    imMOG_arr.append(maskMOG)
    imMOGRGB_arr.append(maskMOGRGB)
    imMOGG_arr.append(maskMOGG)
    imOTSU_arr.append(thresh)
    imOTSUGTH_arr.append(ret)
    imOTSURGB_arr.append(thresh2)
    imOTSUGTH_arr.append(ret2)
    imOTSUG_arr.append(thresh3)
    imOTSUG_arr.append(ret3)
    timestamp_arr.append(timestamp)



for i in range(0, 11):
    t_arr = ['Original', 'RGB Gaussian', 'Gray Gaussian',
             'Histogram ', 'RGB Gaussian', 'Gray Gaussian',
             'MOG', 'RGB MOG', 'Gray MOG',
             'OTSU', 'RGB OTSU', 'Gray OTSU']
    fig, ax = plt.subplots(nrows=3,ncols=4)
    plt.suptitle(timestamp_arr[i])
    ax[0, 0].imshow(imraw_arr[i])
    ax[0, 0].set_title(t_arr[0])
    ax[0, 1] = plt.hist(imraw_arr[i].ravel(),256)
    ax[0, 1].set_title(t_arr[3]+imOTSUTH_arr[i])
    ax[0, 2].imshow(imMOG_arr[i])
    ax[0, 2].set_title(t_arr[6])
    ax[0, 2].imshow(imOTSU_arr[i])
    ax[0, 2].set_title(t_arr[9])

    ax[1, 0].imshow(imGRGB_arr[i])
    ax[1, 0].set_title(t_arr[1])
    ax[1, 1] = plt.hist(imGRGB_arr[i].ravel(), 256)
    ax[1, 1].set_title(t_arr[3] + imOTSURGBTH_arr[i])
    ax[1, 2].imshow(imMOGRGB_arr[i])
    ax[1, 2].set_title(t_arr[7])
    ax[1, 2].imshow(imOTSURGB_arr[i])
    ax[1, 2].set_title(t_arr[10])

    ax[2, 0].imshow(imGG_arr[i])
    ax[2, 0].set_title(t_arr[2])
    ax[2, 1] = plt.hist(imGG_arr[i].ravel(), 256)
    ax[2, 1].set_title(t_arr[3] + imOTSUG_arr[i])
    ax[2, 2].imshow(imMOGG_arr[i])
    ax[2, 2].set_title(t_arr[8])
    ax[2, 2].imshow(imOTSUG_arr[i])
    ax[2, 2].set_title(t_arr[11])

    for j in range(0,3):
        for k in range(0,4):
            ax[j, k].set_yticklabels([])
            ax[j, k].set_xticklabels([])
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")