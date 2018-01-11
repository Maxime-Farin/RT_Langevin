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
'''
nb_capteurs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

energy = np.zeros([len(nb_capteurs), 1], dtype = float)
amplitude_max = np.zeros([len(nb_capteurs), 1], dtype = float)

for cc in range(len(nb_capteurs)):

    filename = "20180111_" + str(nb_capteurs[cc]) + "capteurs_4000.0_7000.0Hz_PSF.mat"

    d = loadmat(filename)
    dist = d['Distance_cm']
    PSF = d['PSF']
    
    energy[cc] = d['energy'][0][0]
    amplitude_max[cc] = d['max_amplitude'][0][0]
    
    #plt.plot(dist[0], PSF[0], label = str(nb_capteurs[cc]))



plt.plot(nb_capteurs, amplitude_max, 'k', linewidth = 2)
plt.xlabel("Number of loudspeakers")
plt.ylabel("Amplitude of the recorded impulse [Counts]")
#plt.legend()
plt.rc('font', size = 18)
plt.show()
'''

'''
plt.xlabel("Distance [cm]")
plt.ylabel("PSF [dB]")
plt.legend()
plt.rc('font', size = 18)
plt.show()
'''

amplitude_max = np.zeros([6, 1], dtype = float)

filename = "20180111_2capteurs_4000.0_7000.0Hz_PSF.mat"
d = loadmat(filename)
dist = d['Distance_cm']
PSF = d['PSF']
amplitude_max[0] = d['max_amplitude'][0][0]
plt.plot(dist[0], PSF[0], label = '2 LS - Parallel to plate')

filename = "20180111_8capteurs_1_8_PSF.mat"
d = loadmat(filename)
dist = d['Distance_cm']
PSF = d['PSF']
amplitude_max[1] = d['max_amplitude'][0][0]
plt.plot(dist[0], PSF[0], label = '8 LS - Parallel to plate')

filename = "20180111_16capteurs_1_24_PSF.mat"
d = loadmat(filename)
dist = d['Distance_cm']
PSF = d['PSF']
amplitude_max[2] = d['max_amplitude'][0][0]
plt.plot(dist[0], PSF[0], label = '16 LS - Parallel to plate')

filename = "20180111_8capteurs_9_16_PSF.mat"
d = loadmat(filename)
dist = d['Distance_cm']
PSF = d['PSF']
amplitude_max[3] = d['max_amplitude'][0][0]
plt.plot(dist[0], PSF[0], label = '8 LS - Normal to plate')

filename = "20180111_16capteurs_9_32_PSF.mat"
d = loadmat(filename)
dist = d['Distance_cm']
PSF = d['PSF']
amplitude_max[4] = d['max_amplitude'][0][0]
plt.plot(dist[0], PSF[0], label = '16 LS - Normal to plate')

filename = "20180111_32capteurs_4000.0_7000.0Hz_PSF.mat"
d = loadmat(filename)
dist = d['Distance_cm']
PSF = d['PSF']
amplitude_max[5] = d['max_amplitude'][0][0]
plt.plot(dist[0], PSF[0], label = '32 LS')

plt.xlabel("Distance [cm]")
plt.ylabel("PSF [dB]")
plt.legend()
plt.rc('font', size = 18)
plt.show()

nb_LS_par = [2, 8, 16]
nb_LS_norm = [8, 16]

plt.plot(nb_LS_par, amplitude_max[0:3], '+', label = 'Parallel to plate')
plt.plot(nb_LS_norm, amplitude_max[3:5], '+', label = 'Normal to plate')
plt.plot([32], amplitude_max[-1], '+', label = '32 LS')

plt.xlabel("Number of loudspeakers")
plt.ylabel("Amplitude of the recorded impulse [Counts]")
plt.legend()
plt.rc('font', size = 18)
plt.show()
