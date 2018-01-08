#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pyaudio
import wave
import numpy as np
import scipy.signal as signal
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from math import *
import time
import pickle


nb_sensors = [2, 4, 6, 8, 10, 12, 14, 16]

for k in range(len(nb_sensors)):
    filename = '20180108_Normal_Plate_' + str(nb_sensors[k]) + 'capteurs_PSF.mat'
    
    d = loadmat(filename)
    dist = d['Distance_cm']
    PSF = d['PSF']
    
    plt.plot(dist[0], PSF[0], label = str(nb_sensors[k]))

plt.xlabel("Distance [cm]")
plt.ylabel("PSF [dB]")
plt.legend()
plt.rc('font', size = 18)
plt.show()