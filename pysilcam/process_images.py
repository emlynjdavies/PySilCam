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
from skimage.filters import threshold_otsu, threshold_adaptive, try_all_threshold
#import sys
#import logging
#from pysilcam.acquisition import Acquire
#from pysilcam.config import PySilcamSettings
#import os
#import psutil
#from shutil import copyfile
#import warnings
#import pandas as pd

# Z:/DATA/SILCAM/250718/config.ini Z:/DATA/SILCAM/RAW/250718 --nomultiproc
config_filename = "Z:/DATA/SILCAM/Working/config.ini"
datapath = "Z:/DATA/SILCAM/Working/RAW250718"
discWrite = False
###################################################

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(each_channel)
def otsu_each(image):
    global_thresh = threshold_otsu(image)
    binary_global = image > global_thresh
    return binary_global

@adapt_rgb(each_channel)
def adaptive_each(image):
    block_size = 35
    global_thresh = threshold_adaptive(image, block_size, offset=10)
    binary_global = image > global_thresh
    return binary_global
####################################################
# Loading config file
# Load the configuration, create settings object
settings = PySilcamSettings(config_filename)

# Print configuration to screen
print('---- CONFIGURATION ----\n')
#print(sys.stdout)

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

# for timestamp, imraw in aqgen:
for i, (timestamp, imc, imraw) in enumerate(bggen):
    x, y, z = imraw.shape
    xc, yc, zc = imc.shape
    print("timestamp ", timestamp)
    print("Image raw shape imraw.shape( ", x,y,z)
    print("Corrected image shape imc.shape( ", xc, yc, zc)
    fig, ax = plt.subplots(nrows=5,ncols=4)  # , figsize=(8, 8)
    plt.suptitle(timestamp)
    #ax = axes.ravel()

    ax[0, 0].imshow(imraw); #plt.colorbar()
    ax[0, 0].set_title('Original ' + str(imraw.shape))  # str(timestamp
    ax[0, 1].hist(imraw.ravel(), 256, [0, 256])
    ax[0, 1].set_title('Original Histogram')
    ax[0, 2].imshow(imc);  # plt.colorbar()
    ax[0, 2].set_title('Corrected ' + str(imc.shape))
    ax[0, 3].hist(imc.ravel(), 256, [0, 256])
    ax[0, 3].set_title('Corrected Histogram')

    gray_image = rgb2gray(imraw)
    ax[1, 0].imshow(gray_image, cmap='gray') #; ax[1].colorbar()
    ax[1, 0].set_title('Gray scale '+ str(gray_image.shape))
    ax[1, 1].hist(gray_image.ravel(), 256, [0, 256])
    ax[1, 1].set_title('Gray scale Histogram')
    gray_image_c = rgb2gray(imc)
    ax[1, 2].imshow(gray_image_c, cmap='gray')  # ; ax[1].colorbar()
    ax[1, 2].set_title('Corrected gray scale '+ str(gray_image_c.shape))
    ax[1, 3].hist(gray_image_c.ravel(), 256, [0, 256])
    ax[1, 3].set_title('Corrected gray scale Histogram')

    #ax[1].imshow(exposure.rescale_intensity(1 - otsu_each(imraw)))
    #ax[1].set_title('otsu')
    adapt = exposure.rescale_intensity(1 - adaptive_each(imraw))
    ax[2, 0].imshow(adapt)
    ax[2, 0].set_title('Adaptive')
    ax[2, 1].hist(adapt.ravel(), 256, [0, 256])
    ax[2, 1].set_title('Adaptive Histogram')
    adaptc = exposure.rescale_intensity(1 - adaptive_each(imc))
    ax[2, 2].imshow(adaptc)
    ax[2, 2].set_title('Corrected adaptive')
    ax[2, 3].hist(adaptc.ravel(), 256, [0, 256])
    ax[2, 3].set_title('Corrected adaptive Histogram')

    ax[3, 0].imshow(imraw[:, :, 2])
    ax[3, 0].set_title('B 2 channel')
    ax[3, 1].hist(imraw[:, :, 2].ravel(), 256, [0, 256])
    ax[3, 1].set_title('B 2 channel Histogram')
    ax[3, 2].imshow(imc[:, :, 2])
    ax[3, 2].set_title('Corrected B 2 channel')
    ax[3, 3].hist(imc[:, :, 2].ravel(), 256, [0, 256])
    ax[3, 3].set_title('Corrected B 2 channel Histogram')

    sob = exposure.rescale_intensity(1 - sobel_each(imraw))
    ax[4, 0].imshow(sob)
    ax[4, 0].set_title('Sobel')
    ax[4, 1].hist(sob.ravel(), 256, [0, 256])
    ax[4, 1].set_title('Sobel Histogram')
    sobc = exposure.rescale_intensity(1 - sobel_each(imc))
    ax[4, 2].imshow(sobc)
    ax[4, 2].set_title('Corrected Sobel')
    ax[4, 3].hist(sobc.ravel(), 256, [0, 256])
    ax[4, 3].set_title('Corrected Sobel Histogram')

    #plt.title(timestamp)
    #fig, ax = try_all_threshold(imraw, figsize=(10, 8), verbose=False)
    for i in range(0,5):
        for j in range(0,4):
            ax[i, j].set_yticklabels([])
            ax[i, j].set_xticklabels([])

    plt.axis('off')
    plt.tight_layout()
    plt.show()

'''    #####  trying out otsu and adaptive thresholding -- binarizing the image////
    global_thresh = threshold_otsu(imraw)
    binary_global = imraw > global_thresh

    block_size = 35
    R_global = threshold_adaptive(imraw[:,:,0], block_size, offset=10)
    binary_adaptiveR = imraw[:,:,0] > R_global

    block_size = 35
    G_global = threshold_adaptive(imraw[:, :, 1], block_size, offset=10)
    binary_adaptiveG = imraw[:,:,1] > G_global

    block_size = 35
    B_global = threshold_adaptive(imraw[:, :, 2], block_size, offset=10)
    binary_adaptiveB = imraw[:,:,2] > B_global

    fig, axes = plt.subplots(ncols=5, figsize=(7, 8))
    ax = axes.ravel()
    #plt.subplots(251)
    #ax0, ax1, ax2,ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes
    #plt.gray()

    #axes[0, 0]
    ax[0].imshow(imraw)
    ax[0].set_title(timestamp)
    #axes[1, 0].hist(imraw.ravel(), 256, [0, 256])
    #axes[1, 0].set_title('Histogram')

    ax[1].imshow(binary_global, cmap=plt.cm.gray)
    ax[1].set_title('Global thresholding')
    #axes[1, 1].hist(binary_global.ravel(), 256, [0, 256])
    #axes[1, 1].set_title('Otsu Histogram')

    ax[2].imshow(binary_adaptiveR, cmap=plt.cm.gray)
    ax[2].set_title('Adaptive thresholding R')
    #axes[1, 2].hist(binary_adaptiveR.ravel(), 256, [0, 256])
    #axes[1, 2].set_title('Adaptive histogram R')

    ax[3].imshow(binary_adaptiveG, cmap=plt.cm.gray)
    ax[3].set_title('Adaptive thresholding G')
    #axes[1, 3].hist(binary_adaptiveG.ravel(), 256, [0, 256])
    #axes[1, 3].set_title('Adaptive histogram G')

    ax[4].imshow(binary_adaptiveB, cmap=plt.cm.gray)
    ax[4].set_title('Adaptive thresholding B')
    #axes[1, 4].hist(binary_adaptiveB.ravel(), 256, [0, 256])
    #axes[1, 4].set_title('Adaptive histogram B')

    for ax in axes:
        ax.axis('off')
    plt.show()
'''





    #stats_all, imbw, saturation = statextract(imraw, settings, timestamp, "", "")
    #print("segmented image ")
    #plt.imshow(imbw);
    #plt.show()
    #image2blackwhite_accurate(imraw, 0.97)
    #fd, hog_image = hog(imraw, orientations=8, pixels_per_cell=(32, 32),
    #                    cells_per_block=(1, 1), visualize=True, multichannel=True)
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    #ax1.axis('off')
    #ax1.imshow(imraw, cmap=plt.cm.gray)
    #ax1.set_title('Input image')

    # Rescale histogram for better display
    #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    #ax2.axis('off')
    #ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    #ax2.set_title('Histogram of Oriented Gradients')
    #plt.show()

    # cropping the image before processing
    ##cimraw = imraw[500:1750, 500:1750]
    ##cimraw = imraw[365:1800, 465:1900]
    #cmin = 470  # min column number
    #cmax = 1900 # max column number
    #rmin = 370  # min row number
    #rmax = 1800 # max row number
    #cimraw = imraw[rmin:rmax, cmin:cmax]   # 1430
    #plt.imshow(cimraw);
    #plt.show()

print ("DONE")