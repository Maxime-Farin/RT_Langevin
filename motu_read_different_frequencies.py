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
Script that plots data recorded in .mat file for the PSF and amplitude/energy of the signal recorded by a pezo on the plate 
when loudspeakers play a signal for different mean frequency annd for different bandwidths
'''

plt.cla()

FREQ0 = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 9000]
FREQ1 = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000]
#FREQ0 = [1000, 3000, 5000, 7000, 9000]
#FREQ1 = [2000, 4000, 6000, 8000, 10000]
Deltaf = [10, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000] # bandwidth
Central_freq = 5500

vector = FREQ0

average_freq = np.zeros([len(vector), 1], dtype = float)
energy = np.zeros([len(vector), 1], dtype = float)
amplitude_max = np.zeros([len(vector), 1], dtype = float)

for ff in range(len(vector)):

    freq0 = FREQ0[ff] #Central_freq - Deltaf[ff] / 2 #FREQ0[ff] # minimum frequency of the chirp
    freq1 = FREQ1[ff] #Central_freq + Deltaf[ff] / 2 #FREQ1[ff] # maximum frequency of the chirp

    # import data from the .mat files
    filename = '20180111_sensor_in_air_' + str(freq0) + '_' + str(freq1) + 'Hz.mat' 
    d = loadmat(filename)
    #dist = d['Distance_cm']
    #PSF = d['PSF']
    
    energy[ff] = d['energy'][0][0] # energy of the recorded signal (trapz(signal**2)
    amplitude_max[ff] = d['max_amplitude'][0][0] # maximum amplitude of the recorded signal
    
    average_freq[ff] = float(freq0 + freq1) / 2 # average frequency of the played signal
    #plt.plot(dist[0], PSF[0], label = str(average_freq[ff]))
    #plt.plot(dist[0], PSF[0], label = str(Deltaf[ff]))
# Uncomment below to plot energy/amplitude vs frequency

plt.plot(average_freq, amplitude_max, 'k', linewidth = 2)
plt.xlabel("Mean Frequency [Hz]")
#plt.plot(Deltaf, amplitude_max, 'k', linewidth = 2)
#plt.xlabel("Bandwidth [Hz]")
plt.ylabel("Amplitude of the recorded impulse [Counts]")
#plt.legend()
plt.rc('font', size = 18)
plt.show()    


# Uncomment below to plot PSF vs Distance as function of frequency
'''
plt.xlabel("Distance [cm]")
plt.ylabel("PSF [dB]")
plt.legend()
plt.rc('font', size = 18)
plt.show()
'''