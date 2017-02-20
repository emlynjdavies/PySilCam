# -*- coding: utf-8 -*-
import sys
import time
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cProfile
import pstats
from io import StringIO
from pysilcam import __version__
from pysilcam.acquisition import acquire
from pysilcam.background import backgrounder
import pysilcam.process
from pysilcam.process import statextract
import pysilcam.postprocess as sc_pp
from pysilcam.config import load_config, PySilcamSettings


def silcam_acquire():
    '''Aquire images from the SilCam

    Usage:
      silcam-acquire liveview
      silcam-acquire process <configfile>
      silcam-acquire -h | --help
      silcam-acquire --version

    Arguments:
        liveview    Display acquired images
        process     Process acquired images in real time

    Options:
      -h --help     Show this screen.
      --version     Show version.
    '''
    args = docopt(silcam_acquire.__doc__, version='PySilCam {0}'.format(__version__))
    #print('Type \'silcam-acquire -h\' for help')

    if args['process']:
        pr = cProfile.Profile()
        pr.enable()
        s = StringIO()
        sortby = 'cumulative'
        silcam_process_realtime(args['<configfile>'])
        pr.disable()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
#        print(s.getvalue())
        ps.dump_stats('process_profile.cprof')

    elif args['liveview']:
        plt.ion()
        fig, ax = plt.subplots()
        t1 = time.time()
        for i, img in enumerate(acquire()):
            t2 = time.time()
            aq_freq = np.round(1.0/(t2 - t1), 1)
            print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, aq_freq))
            t1 = t2
            ax.imshow(img[:,:,0], cmap=plt.cm.gray)
            plt.draw()
            plt.pause(0.05)

    else:
        t1 = time.time()
        for i, img in enumerate(acquire()):
            t2 = time.time()
            aq_freq = np.round(1.0/(t2 - t1), 1)
            print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, aq_freq))
            t1 = t2



def silcam_process_realtime(config_filename):
    '''Run processing of SilCam images in real time'''

    print('PYSILCAM REALTIME MODE')
    print('======================\n')

    #Load the configuration, create settings object
    conf = load_config(config_filename)
    conf.write(sys.stdout)
    print('----------------------\n')
    settings = PySilcamSettings(conf)

    #Initialize the image acquisition generator
    aqgen = acquire()

    #Get number of images to use for background correction from config
    n = settings.Background.num_images

    d50_ts = []

    for i, imc in enumerate(backgrounder(n, aqgen)):
        print('PROCESSING....')
        start_time = time.clock()
        stats = statextract(imc, i, settings.Process)
        proc_time = time.clock() - start_time
        print('PROCESSING DONE in', proc_time, 'sec.')

        if stats is not np.nan:
            print('data has arrived!')
        stats = sc_pp.filter_stats(stats, settings.PostProcess)
        d50 = sc_pp.d50_from_stats(stats)
        print('d50:', d50)

        d50_ts.append(d50)

        plt.cla()
        plt.plot(d50_ts,'.')
        plt.draw()
        plt.pause(0.05)

        if (i == 10):
            plt.show()
            break

    
def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
