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
from motu import *

'''
Script that compute the amplitude spectrum of a compressed impulsion played by a loudspeaker 
in order to evaluate the frequency response of the loudspeaker.
'''


m = motu() # define a class motu named m

try: 


    freq0 = 100
    freq1 = 25000
    duree = 3

    ChannelsOut = [28]
    ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
    ChannelsIn = [13] #capteur piezo sur la plaque
    ChannelsIn = [c - 1 for c in ChannelsIn]

    # emits a chirp from each ChannelsOut successively and record it on ChannelIn
    impulse = np.zeros([ceil(m.RATE*duree), len(ChannelsOut)])

    impulse = m.ChirpRec(ChannelsIn, ChannelsOut[0], freq0 = freq0, freq1 = freq1, duree = duree)[:, 0] # reponse impulsionnelle
    # if one want a chirp (for example to calibrate the response between source and receiver)

    time_impulse = np.arange(0.0, duree, 1.0 / float(m.RATE))


    plt.plot(time_impulse, impulse)
    plt.xlabel("Time [s]")
    plt.ylabel("Counts")
    plt.rc('font', size = 18)
    plt.show()
      

    ind1 = ceil(0.214*m.RATE)
    ind2 = ceil(0.217*m.RATE)


    plt.plot(time_impulse[ind1:ind2] - time_impulse[ind1], impulse[ind1:ind2])
    plt.xlabel("Time [s]")
    plt.ylabel("Impulse [counts]")
    plt.title("Impulse sent by " + str(ChannelsOut[0] + 1) + ", recorded on Micro")
    plt.rc('font', size = 18)
    plt.show()

    # Amplitude spectrum
    xF = np.fft.rfft(impulse[ind1:ind2])

    freq = np.linspace(0, m.RATE/2, len(xF))

    # Plot Amplitude spectrum
    plt.plot(freq, abs(xF), 'k')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude Spectrum [Counts]")
    plt.title("Impulse sent by " + str(ChannelsOut[0] + 1) + ", recorded on Micro")
    plt.rc('font', size = 18)
    plt.xlim([0, 25000])
    plt.show()#block = False) 

    # Compute Average frequency
    # mean_freq = ceil(np.trapz(abs(xF[:,0])*freq) / np.trapz(abs(xF[:,0])))
    # print(mean_freq)

    savefilename = "20180110_Frequency_response_loudspeaker_" + str(ChannelsOut[0])
    data = {'freq': freq, 'Amplitude_Spectrum': xF, 'time': time_impulse[ind1:ind2] - time_impulse[ind1], 'impulse': impulse[ind1:ind2]}
    savemat(savefilename, data)
    
    m.stream.stop_stream()
    m.stream.close()
    m.p.terminate()
    
except:
    m.stream.stop_stream()
    m.stream.close()
    m.p.terminate()