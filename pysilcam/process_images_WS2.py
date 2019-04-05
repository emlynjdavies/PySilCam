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
laplace_arr = []
bw_arr = []
distTransf_arr = []
peaks_arr = []
markers_arr = []
final_arr = []
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG2()

for timestamp, imraw in aqgen:
    ### imraw -- original image
    imraw_arr.append(imraw)
    gray = cv.cvtColor(imraw, cv.COLOR_RGB2GRAY)
    maskMOG = subtractorMOG.apply(imraw)
    ### maskMOG -- background subtracted
    imMOG_arr.append(maskMOG)

    #####################################################################
    #### SHARPEN THE MASKMOG GENERATED BACKGROUND SUBTRACTED IMAGE
    # Create a kernel that we will use to sharpen our image
    # an approximation of second derivative, a quite strong kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    # so the possible negative number will be truncated
    imgLaplacian = cv.filter2D(maskMOG, cv.CV_32F, kernel)
    sharp = np.float32(maskMOG)
    imgResult = sharp - imgLaplacian
    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    #### imageResult -- Laplacian filter  New Sharped Image
    laplace_arr.append(imgResult)

    # Create binary image from source image
    bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    ## bw  -- Binary Image
    bw_arr.append(bw)

    # Perform the distance transform algorithm
    dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    ### dist -- Distance Transform Image
    distTransf_arr.append(dist)

    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
    # Dilate a bit the dist image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)
    ### dist -- Peaks
    peaks_arr.append(dist)

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype('uint8')
    # Find total markers
    _, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    ### markers*10000 -- Markers
    markers_arr.append(markers)

    # Perform the watershed algorithm
    cv.watershed(imgResult, markers)
    # mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)
    # uncomment this if you want to see how the mark
    # image looks like at that point
    # cv.imshow('Markers_v2', mark)
    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :] = colors[index - 1]
    # Visualize the final image
    ### dst -- Final Result
    final_arr.append(dst)





    #####################################################################

    timestamp_arr.append(timestamp)
    t_arr = ['Original', 'MOG', 'Laplacian',
             'Binary', 'Dist Transform', 'Peaks',
             'Markers', 'Final Result'
             ]


for i in range(0, 11):

    fig, ax = plt.subplots(nrows=3, ncols=3)
    ax[0, 0].imshow(imraw_arr[i])
    ax[0, 0].set_title(t_arr[0])

    ax[0, 1].imshow(bw_arr[i])
    ax[0, 1].set_title(t_arr[3])

    ax[0, 2].imshow(markers_arr[i])
    ax[0, 2].set_title(t_arr[6])

    ax[1, 0].imshow(imMOG_arr[i])
    ax[1, 0].set_title(t_arr[1])

    ax[1, 1].imshow(distTransf_arr[i])
    ax[1, 1].set_title(t_arr[4])

    ax[1, 2].imshow(imraw_arr[i])
    ax[1, 2].set_title(t_arr[7])

    ax[2, 0].imshow(laplace_arr[i])
    ax[2, 0].set_title(t_arr[2])

    ax[2, 1].imshow(peaks_arr[i])
    ax[2, 1].set_title(t_arr[5])

    for j in range(0,3):
        for k in range(0,3):
            ax[j, k].set_yticklabels([])
            ax[j, k].set_xticklabels([])
    plt.suptitle(timestamp_arr[i])
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")