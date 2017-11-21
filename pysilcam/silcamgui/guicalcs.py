import pysilcam.__main__ as psc
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QWidget,
QAction, QTabWidget,QVBoxLayout, QFileDialog)
import os
from pysilcam.config import load_config, PySilcamSettings
import pysilcam.oilgas as scog
import numpy as np
import pysilcam.postprocess as sc_pp
import pandas as pd

def get_data(self):
    try:
        stats = self.q.get(timeout=0.1)
    except:
        stats = None
    return stats


def extract_stats_im(guidata):
    imc = guidata['imc']
    del guidata['imc']
    stats = pd.DataFrame.from_dict(guidata)
    return stats, imc


class ProcThread(Process):

    def __init__(self, datadir):
        super(ProcThread, self).__init__()
        self.q = Queue()
        self.info = 'ini done'
        self.datadir = datadir
        self.configfile = ''
        self.settings = ''
        self.rts = ''


    def run(self):
        psc.silcam_process(self.configfile, self.datadir, gui=self.q)
        #psc.silcam_sim(self.datadir, self.q)


    def go(self):
        self.start()


    def stop_silcam(self):
        if self.is_alive():
            self.terminate()
            self.info = 'termination sent'
        else:
            self.info = 'nothing to terminate'


    def plot(self):
        infostr = 'waiting to plot'
        if self.rts == '':
            self.rts = scog.rt_stats(self.settings)


        if self.is_alive():
            guidata = get_data(self)
            if not guidata == None:
                stats, imc = extract_stats_im(guidata)
                timestamp = np.max(stats['timestamp'])

                try:
                    self.rts.stats = self.rts.stats().append(stats)
                except:
                    self.rts.stats = self.rts.stats.append(stats)
                self.rts.update()
                filename = os.path.join(self.settings.General.datafile,
                    'OilGasd50.csv')
                self.rts.to_csv(filename)

                dias, vd_oil = sc_pp.vd_from_stats(self.rts.oil_stats,
                    self.settings.PostProcess)
                dias, vd_gas = sc_pp.vd_from_stats(self.rts.gas_stats,
                    self.settings.PostProcess)

                #infostr = data['infostr']
                infostr = 'got data'

                plt.clf()
                plt.cla()


                plt.subplot(2,2,1)
                plt.cla()
                plt.plot(dias, vd_oil ,'r')
                plt.plot(dias, vd_gas ,'b')
                plt.xscale('log')
                plt.xlim((10, 10000))
                plt.ylabel('Volume Concentration [uL/L]')
                plt.xlabel('Diamter [um]')

                plt.subplot(2,2,3)
                ttlstr = (
                        'Oil d50: {:0.0f}um'.format(self.rts.oil_d50) + '\n' +
                        'Gas d50: {:0.0f}um'.format(self.rts.gas_d50) + '\n'
                        )
                plt.title(ttlstr)
                plt.axis('off')

                plt.subplot(1,2,2)
                ttlstr = ('Image time: ' +
                    str(timestamp))
                plt.cla()
                plt.imshow(imc)
                plt.axis('off')
                plt.title(ttlstr)

                plt.tight_layout()
            #except:
            #    infostr = 'failed to get data from process'
            #    print('failed to get data from process')

            while not self.q.empty():
                self.q.get_nowait()

            self.info = infostr


    def silcview(self):
        self.raw()
        files = [os.path.join(self.datadir, f) for f in
                sorted(os.listdir(self.datadir))
                if f.endswith('.bmp')]
        import pygame
        import time
        pygame.init()
        info = pygame.display.Info()
        size = (int(info.current_h / (2048/2448))-100, info.current_h-100)
        screen = pygame.display.set_mode(size)
        font = pygame.font.SysFont("monospace", 20)
        c = pygame.time.Clock()
        zoom = False
        counter = -1
        direction = 1 # 1=forward 2=backward
        pause = False
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        while True:
            if pause:
                event = pygame.event.wait()
                if event.type == 12:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        zoom = np.invert(zoom)
                    if event.key == pygame.K_LEFT:
                        direction = -1
                    if event.key == pygame.K_RIGHT:
                        direction = 1
                    if event.key == pygame.K_p:
                        pause = np.invert(pause)
                    else:
                        continue
                    pygame.time.wait(100)

            counter += direction
            counter = np.max([counter, 0])
            counter = np.min([len(files)-1, counter])
            c.tick(15) # restrict to 15Hz
            f = files[counter]

            if not (counter == 0 | counter==len(files)-1):
                im = pygame.image.load(os.path.join(self.datadir, f)).convert()

            if zoom:
                label = font.render('ZOOM [F]: ON', 1, (255, 255, 0))
                im = pygame.transform.scale2x(im)
                screen.blit(im,(-size[0]/2,-size[1]/2))
            else:
               im = pygame.transform.scale(im, size)
               screen.blit(im,(0,0))
               label = font.render('ZOOM [F]: OFF', 1, (255, 255, 0))
            screen.blit(label,(0, size[1]-20))

            if direction==1:
                dirtxt = '>>'
            elif direction==-1:
                dirtxt = '<<'
            if pause:
                dirtxt = 'PAUSED ' + dirtxt
            label = font.render('DIRECTION [<-|->|p]: ' + dirtxt, 1, (255,255,0))
            screen.blit(label, (0, size[1]-40))

            if counter == 0:
                label = font.render('FIRST IMAGE', 1, (255,255,0))
                screen.blit(label, (0, size[1]-60))
            elif counter == len(files)-1:
                label = font.render('LAST IMAGE', 1, (255,255,0))
                screen.blit(label, (0, size[1]-60))


            timestamp = pd.to_datetime(
                    os.path.splitext(os.path.split(f)[-1])[0][1:])


            pygame.display.set_caption('raw image replay:' + os.path.split(f)[0])#, icontitle=None)
            label = font.render(str(timestamp), 20, (255, 255, 0))
            screen.blit(label,(0,0))
            label = font.render('Display FPS:' + str(c.get_fps()),
                    1, (255, 255, 0))
            screen.blit(label,(0,20))

            for event in pygame.event.get():
                if event.type == 12:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        zoom = np.invert(zoom)
                    if event.key == pygame.K_LEFT:
                        direction = -1
                    if event.key == pygame.K_RIGHT:
                        direction = 1
                    if event.key == pygame.K_p:
                        pause = np.invert(pause)
                        direction = 0



            pygame.display.flip()

        pygame.quit()

        print(self.datadir)


    def load_settings(self, configfile):
        conf = load_config(configfile)
        self.settings = PySilcamSettings(conf)
