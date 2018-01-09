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

nb_capteurs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

for cc in range(len(nb_capteurs)):

for k in range(len(nb_sensors)):
    filename = "20180109_" + str(nb_capteurs[cc]) + "capteurs_" + str(freq0) + "_" + str(freq1) + "Hz.mat"
    
    d = loadmat(filename)
    dist = d['Distance_cm']
    PSF = d['PSF']
    
    energy[ff] = d['energy'][0][0]
    amplitude_max[ff] = d['max_amplitude'][0][0]
    
    average_freq[ff] = float(freq0 + freq1) / 2
    plt.plot(dist[0], PSF[0], label = str(average_freq[ff]))
    
    plt.plot(dist[0], PSF[0], label = str(nb_sensors[k]))

plt.xlabel("Distance [cm]")
plt.ylabel("PSF [dB]")
plt.legend()
plt.rc('font', size = 18)
plt.show()