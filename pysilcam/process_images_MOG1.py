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
imMOGSegO_arr = []
imMOGSeg2_arr = []
imMOGSegO2_arr = []

imMASeg_arr = []
imMASegO_arr = []
imMASeg2_arr = []
imMASegO2_arr = []
timestamp_arr = []

aq=Acquire(USE_PYMBA=False)   # USE_PYMBA=realtime
print ("Acquiring images...")
aqgen=aq.get_generator(datapath,writeToDisk=discWrite,
                       camera_config_file=config_filename)
subtractorMOG = cv.createBackgroundSubtractorMOG2(history=5, varThreshold=25, detectShadows=False)

for timestamp, imraw in aqgen:
    maskMOG = subtractorMOG.apply(imraw)
    maskMOG2 = np.copy(maskMOG)
    imraw_arr.append(np.copy(imraw))
    imMOG_arr.append(np.copy(maskMOG))
    timestamp_arr.append(timestamp)
    # Finding foreground area
    # closing operation
    ret, thresh = cv.threshold(maskMOG, 0, 255,
                                 cv.THRESH_BINARY_INV +
                                 cv.THRESH_OTSU)
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
    ret, fg = cv.threshold(dist_transform, 0.7
                             * dist_transform.max(), 255, 0)
    ###
    ####  WATERSHED ALGORITHM #################################
    # Marker labelling
    fg = np.uint8(fg)
    unknown = cv.subtract(bg,fg)
    ret, markers = cv.connectedComponents(fg)
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markersO = markers

    new_gray = cv.cvtColor(maskMOG, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(new_gray, markers)
    maskMOG[markers == -1] = 255
    imMOGSeg_arr.append(maskMOG)

    markersO = cv.watershed(new_gray, markersO)
    imraw[markersO == -1] = [255,0,0]
    imMOGSegO_arr.append(imraw)

    #####################################################################
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

    new_gray = cv.cvtColor(imgResult, cv.COLOR_GRAY2BGR)
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
    imMOGSeg2_arr.append(dst)

    ### WATERSHED ON THE ORIGINAL IMAGE #############################

    cv.watershed(imraw, markersO)

    # mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markersO.astype('uint8')
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
    dstO = np.zeros((markersO.shape[0], markersO.shape[1], 3), dtype=np.uint8)
    print('dstO.shape = np.zeros((markersO.shape[0], markersO.shape[1],3), dtype=np.uint8) ', dst.shape)
    # Fill labeled objects with random colors
    for i in range(markersO.shape[0]):
        for j in range(markersO.shape[1]):
            index = markersO[i, j]
            if index > 0 and index <= len(contours):
                dstO[i, j, :] = colors[index - 1]
    print('dstO.shape -- final ', dstO.shape)
    # Visualize the final image
    ### dst -- Final Result
    imMOGSegO2_arr.append(dstO)



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
    imc2 = np.copy(imc)
    imMA_arr.append(np.copy(imc))
    imc3 = cv.cvtColor(imc, cv.COLOR_BGR2GRAY)

    _, inv = cv.threshold(imc3, 127, 255, cv.THRESH_BINARY_INV)

    # Finding foreground area
    # closing operation
    ret, thresh = cv.threshold(inv, 0, 255,
                               cv.THRESH_BINARY_INV +
                               cv.THRESH_OTSU)
    ###
    # Noise removal using Morphological
    # closing operation
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE,
                              kernel, iterations=2)
    # Background area using Dialation
    bg = cv.dilate(opening, kernel, iterations=3)
    # Finding foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, fg = cv.threshold(dist_transform, 0.7
                           * dist_transform.max(), 255, 0)
    ###
    ####  WATERSHED ALGORITHM #################################
    # Marker labelling
    fg = np.uint8(fg)
    unknown = cv.subtract(bg, fg)
    ret, markers = cv.connectedComponents(fg)
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markersO = markers

    markers = cv.watershed(inv, markers)
    imc[markers == -1] = 255
    imMASeg_arr.append(imc)

    markersO = cv.watershed(inv, markersO)
    imraw[markersO == -1] = [255, 0, 0]
    imMASegO_arr.append(imraw)

    #####################################################################
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
    imgLaplacian = cv.filter2D(imc2, cv.CV_32F, kernel)
    print('imgLaplacian.shape ', imgLaplacian.shape)  # (2050, 2448)
    sharp = np.float32(imc2)
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
    bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
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

    #new_gray = cv.cvtColor(imgResult, cv.COLOR_GRAY2BGR)
    cv.watershed(imgResult, markers)

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
    imMASeg2_arr.append(dst)

    ### WATERSHED ON THE ORIGINAL IMAGE #############################

    cv.watershed(imc2, markersO)

    # mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markersO.astype('uint8')
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
    dstO = np.zeros((markersO.shape[0], markersO.shape[1], 3), dtype=np.uint8)
    print('dstO.shape = np.zeros((markersO.shape[0], markersO.shape[1],3), dtype=np.uint8) ', dst.shape)
    # Fill labeled objects with random colors
    for i in range(markersO.shape[0]):
        for j in range(markersO.shape[1]):
            index = markersO[i, j]
            if index > 0 and index <= len(contours):
                dstO[i, j, :] = colors[index - 1]
    print('dstO.shape -- final ', dstO.shape)
    # Visualize the final image
    ### dst -- Final Result
    imMASegO2_arr.append(dstO)

for i in range(0, 15):
    fig, ax = plt.subplots(nrows=4, ncols=3)
    plt.suptitle(timestamp_arr[i])
    ax[0, 0].imshow(imraw_arr[i])
    ax[0, 0].set_title('Original')
    ax[0, 1].imshow(imMOG_arr[i])
    ax[0, 1].set_title('MOG' + str(imMOG_arr[i].shape))
    ax[1, 1].imshow(imMOGSeg_arr[i])
    ax[1, 1].set_title('MOG Seg ' + str(imMOGSeg_arr[i].shape))
    ax[1, 0].imshow(imMOGSegO_arr[i])
    ax[1, 0].set_title('MOG Seg Org' + str(imMOGSegO_arr[i].shape))
    ax[2, 1].imshow(imMOGSeg2_arr[i])
    ax[2, 1].set_title('MOG Seg w Laplace ' + str(imMOGSeg2_arr[i].shape))
    ax[3, 1].imshow(imMOGSegO2_arr[i])
    ax[3, 1].set_title('MOG Seg L Org ' + str(imMOGSegO2_arr[i].shape))


    if i > 4:
        ax[0, 2].imshow(imMA_arr[i-5])
        ax[0, 2].set_title('Moving Avg ' + str(imMA_arr[i-5].shape))
        ax[2, 1].imshow(imMASeg_arr[i - 5])
        ax[2, 1].set_title('Moving Avg Seg ' + str(imMASeg_arr[i - 5].shape))
        ax[2, 0].imshow(imMASegO_arr[i - 5])
        ax[2, 0].set_title('MA Seg Org' + str(imMASegO_arr[i - 5].shape))
        ax[2, 2].imshow(imMASeg2_arr[i - 5])
        ax[2, 2].set_title('Moving Average Seg Laplace ' + str(imMASeg2_arr[i - 5].shape))
        ax[3, 2].imshow(imMASegO2_arr[i - 5])
        ax[3, 2].set_title('Moving Average Seg Laplace ' + str(imMASegO2_arr[i - 5].shape))

    for j in range(0, 4):
        for k in range(0, 3):
            ax[j, k].set_yticklabels([])
            ax[j, k].set_xticklabels([])

    plt.axis('off')
    plt.tight_layout()
    plt.show()

print ("DONE")