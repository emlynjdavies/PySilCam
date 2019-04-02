# -*- coding: utf-8 -*-
import sys
import time
import datetime
import logging
from docopt import docopt
import numpy as np
from pysilcam import __version__
from pysilcam.acquisition import Acquire
from pysilcam.background import backgrounder
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
config_filename = "Z:/DATA/SILCAM/Working/config.ini"
datapath = "Z:/DATA/SILCAM/Working/RAW250718"
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
aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)

#Get number of images to use for background correction from config
print('* Initializing background image handler')
bggen = backgrounder(3, aqgen,
                     bad_lighting_limit = None,
                     real_time_stats=False)
count = 0
for timestamp, imraw in aqgen:
# for i, (timestamp, imc, imraw) in enumerate(bggen):
    if count == 0:
        imraw0 = imraw
        first_gray = cv.cvtColor(imraw0, cv.COLOR_RGB2GRAY)

    x, y, z = imraw.shape
    print("timestamp ", timestamp)
    print("Image raw shape imraw.shape( ", x,y,z)
    gray_frame = cv.cvtColor(imraw, cv.COLOR_RGB2GRAY)
    difference = cv.absdiff(first_gray, gray_frame)
    _, difference = cv.threshold(difference, 25, 255, cv.THRESH_BINARY)

    r = 2
    c = 3
    t_arr = ['Background Image','Original Image','Difference']
    fig, ax = plt.subplots(nrows=r,ncols=c)  # , figsize=(8, 8)
    plt.suptitle(timestamp)
    #ax = axes.ravel()

    #ax[0, 0].imshow(imraw); #plt.colorbar()
    #ax[0, 0].set_title('Original ' + str(imraw.shape))  # str(timestamp
    #ax[0, 1].hist(imraw.ravel(), 256, [0, 256])
    #ax[0, 1].set_title('Original Histogram')
    #ax[0, 2].imshow(imc);  # plt.colorbar()
    #ax[0, 2].set_title('Corrected ' + str(imc.shape))
    #ax[0, 3].hist(imc.ravel(), 256, [0, 256])
    #ax[0, 3].set_title('Corrected Histogram')

    for i in range(0, 1):
        ax[i, 0].imshow(imraw0)
        ax[i, 0].set_title(t_arr[0])
        ax[i, 1].imshow(imraw)
        ax[i, 1].set_title(t_arr[1])
        ax[i, 2].imshow(difference)
        ax[i, 2].set_title(t_arr[2])
        for j in range(0, c):
            ax[i, j].set_yticklabels([])
            ax[i, j].set_xticklabels([])

    plt.axis('off')
    plt.tight_layout()
    plt.show()


print ("DONE")