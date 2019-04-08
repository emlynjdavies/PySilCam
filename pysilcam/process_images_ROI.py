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
from scipy.ndimage import label
# Z:/DATA/SILCAM/250718/config.ini Z:/DATA/SILCAM/RAW/250718 --nomultiproc
#config_filename = "Z:/DATA/SILCAM/Working/config.ini"
config_filename = "/mnt/DATA/SILCAM/Working/config.ini"
#datapath = "Z:/DATA/SILCAM/Working/RAW250718"
datapath = "/mnt/DATA/SILCAM/Working/RAW250718"
discWrite = False
###################################################
def segment_on_dt(a, img):
    border = cv.dilate(img, None, iterations=5)
    border = border - cv.erode(border, None)

    dt = cv.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv.threshold(dt, 180, 255, cv.THRESH_BINARY)  # 0 -> 180
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl
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
imMOG_INV_arr = []
imMOGSeg_arr = []
imMOGSeg2_arr = []
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG2(history=5, varThreshold=25, detectShadows=False)

for timestamp, imraw in aqgen:
    maskMOG = subtractorMOG.apply(imraw)
    _, maskMOG_INV = cv.threshold(maskMOG, 127, 255, cv.THRESH_BINARY_INV)
    imraw_arr.append(imraw)
    maskMOG_cp = np.copy(maskMOG)
    maskMOG2 = np.copy(maskMOG_INV)
    imMOG_arr.append(maskMOG_cp)
    imMOG_INV_arr.append(maskMOG2)
    timestamp_arr.append(timestamp)
    # Finding foreground area
    # closing operation
    ret, thresh = cv.threshold(maskMOG_INV, 0, 255,
                                 cv.THRESH_BINARY +
                                 cv.THRESH_OTSU)

    ###
    # Noise removal using Morphological
    # closing operation
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # find contours
    _, ctrs, _ = cv.findContours(sure_fg.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    new_gray = cv.cvtColor(maskMOG_INV, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(new_gray, markers)
    maskMOG_INV[markers == -1] = 0
    imMOGSeg2_arr.append(maskMOG_INV)


    new_imraw = np.copy(imraw)
    new_imraw[markers == -1] = 255
    new_imraw[markers] = 255
    imMOGSeg_arr.append(new_imraw)


    '''    #####################################################################
    ####  WATERSHED ALGORITHM V2 ########################################
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
    imgLaplacian = cv.filter2D(maskMOG2, cv.CV_32F, kernel)
    print('imgLaplacian.shape ', imgLaplacian.shape)  # (2050, 2448)
    sharp = np.float32(maskMOG2)
    print('sharp.shape ', sharp.shape)  # (2050, 2448)
    imgResult = sharp - imgLaplacian
    print('imgResult.shape = sharp - imgLaplacian ', imgResult.shape)  # (2050, 2448)
    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    print('imgResult.shape = np.clip(imgResult, 0, 255) ', imgResult.shape)  # (2050, 2448)
    imgResult = imgResult.astype('uint8')
    print('imgResult.shape = imgResult.astype(uint8)', imgResult.shape)  # (2050, 2448)
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    print('imgLaplacian.shape = np.clip(imgLaplacian,0,255) ', imgLaplacian.shape)  # (2050, 2448)
    imgLaplacian = np.uint8(imgLaplacian)
    print('imgLaplacian.shape = np.uint8(imgLaplacian) ', imgLaplacian.shape)  # (2050, 2448)
    #### imageResult -- Laplacian filter  New Sharped Image

    # Create binary image from source image
    bw = imgResult  # cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
    print('bw.shape ', bw.shape)
    _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print('bw.shape = cv.threshold(bw, 40, 255, BINARY|OTSU)', bw.shape)
    ## bw  -- Binary Image

    # Perform the distance transform algorithm
    dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
    print('dist.shape = cv.distanceTransform(bw, cv.DIST_L2,3) ', dist.shape)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    print('dist.shape ', dist.shape)
    ### dist -- Distance Transform Image

    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
    print('dist.shape = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY) ', dist.shape)
    # Dilate a bit the dist image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    print('kernel1.shape ', kernel1.shape)
    dist = cv.dilate(dist, kernel1)
    print('dist.shape = cv.dilate(dist, kernel1) ', dist.shape)
    ### dist -- Peaks

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype('uint8')
    print('dist_8u.shape = dist.astype(uint8) ', dist_8u.shape)
    # Find total markers
    _, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)

    new_gray = cv.cvtColor(maskMOG2, cv.COLOR_GRAY2BGR)
    cv.watershed(new_gray, markers)

    # mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)
    print('mark.shape ', mark.shape)
    # uncomment this if you want to see how the mark
    # image looks like at that point
    # cv.imshow('Markers_v2', mark)
    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    print('dst.shape = np.zeros((markers.shape[0], markers.shape[1],3), dtype=np.uint8) ', dst.shape)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :] = colors[index - 1]
    print('dst.shape -- final ', dst.shape)
    # Visualize the final image
    ### dst -- Final Result
'''


for i in range(0, 15):
    fig, ax = plt.subplots(nrows=5)
    plt.suptitle(timestamp_arr[i])
    ax[0].imshow(imraw_arr[i])
    ax[0].set_title('Original')
    ax[1].imshow(imMOG_arr[i])
    ax[1].set_title('MOG hist=5 thres=25 ' + str(imMOG_arr[i].shape))
    ax[2].imshow(imMOG_INV_arr[i])
    ax[2].set_title('MOG Inverted ' + str(imMOG_INV_arr[i].shape))
    ax[3].imshow(imMOGSeg_arr[i])
    ax[3].set_title('MOG Seg ' + str(imMOGSeg_arr[i].shape))
    ax[4].imshow(imMOGSeg2_arr[i])
    ax[4].set_title('MOG Inverted Seg ' + str(imMOGSeg2_arr[i].shape))

    for j in range(0, 5):
        ax[j].set_yticklabels([])
        ax[j].set_xticklabels([])

    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")