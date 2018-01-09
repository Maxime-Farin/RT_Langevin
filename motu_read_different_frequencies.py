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

FREQ0 = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 9000]
FREQ1 = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000]

average_freq = np.zeros([len(FREQ0), 1], dtype = float)
energy = np.zeros([len(FREQ0), 1], dtype = float)
amplitude_max = np.zeros([len(FREQ0), 1], dtype = float)

for ff in range(len(FREQ0)):

    filename = '20180109_32capteurs_' + str(FREQ0[ff]) + '_' + str(FREQ1[ff]) + 'Hz_PSF.mat'
    
    d = loadmat(filename)
    dist = d['Distance_cm']
    PSF = d['PSF']
    
    energy[ff] = d['energy'][0][0]
    amplitude_max[ff] = d['max_amplitude'][0][0]
    
    average_freq[ff] = float(FREQ0[ff] + FREQ1[ff]) / 2
    plt.plot(dist[0], PSF[0], label = str(average_freq[ff]))
'''
plt.plot(average_freq, energy, 'k', linewidth = 2)
plt.xlabel("Mean Frequency [Hz]")
plt.ylabel("Energy of the recorded impulse [Counts]")
#plt.legend()
plt.rc('font', size = 18)
plt.show()    
'''
plt.xlabel("Distance [cm]")
plt.ylabel("PSF [dB]")
plt.legend()
plt.rc('font', size = 18)
plt.show()
