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

m = motu() # define a class motu named m
plt.cla() # clear axis

Central_freq = 5500
freq0 = Central_freq - 5000 / 2 #FREQ0[ff]
freq1 = Central_freq + 5000 / 2 #FREQ1[ff]

nb_capteurs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

for cc in range(len(nb_capteurs)):

    ChannelsOut = list(range(nb_capteurs[cc]))
    #ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
    ChannelsIn = [9] #capteur piezo sur la plaque
    ChannelsIn = [c - 1 for c in ChannelsIn]

    # emits a chirp from each ChannelsOut successively and record it on ChannelIn
    impulse = np.zeros([ceil(m.RATE*m.dureeImpulse), len(ChannelsOut)], dtype = np.int32)
    
    for k in range(len(ChannelsOut)):
        impulse[:, k] = m.ChirpRec(ChannelsIn, ChannelsOut[k], freq0 = freq0, freq1 = freq1)[:, 0] # reponse impulsionnelle
        time.sleep(0.5)
    # if one want a chirp (for example to calibrate the response between source and receiver)
    
    time_impulse = np.arange(0.0, m.dureeImpulse, 1.0 / float(m.RATE))
    
    '''
    plt.plot(time_impulse, impulse)
    plt.xlabel("Time [s]")
    plt.ylabel("Counts")
    plt.show()
    '''

    # use the fact that the sound path is reversible h(-t) = h(t)
    TRSignal = impulse[: : -1, :] / np.abs(impulse).max() # signal is reserved temporally and normalized to 1
    time_result = np.arange(0.0, m.RECORD_SECONDS, 1.0 / float(m.RATE)) 
    
    
    savefilename = "20180109_" + str(nb_capteurs[cc]) + "capteurs_" + str(freq0) + "_" + str(freq1) + "Hz"
    ChannelsPlate = [9, 10, 11, 12, 13, 14]
    ChannelsPlate = [c - 1 for c in ChannelsPlate]

    max_amplitude = np.zeros([len(ChannelsPlate), 1], dtype = np.float)
    energy = np.zeros([len(ChannelsPlate), 1], dtype = np.float)
    #mean_freq = np.zeros([len(ChannelsPlate), 1], dtype = np.float) 
    
    for k in range(len(ChannelsPlate)):
        result = m.PlayAndRec(ChannelsPlate[k], ChannelsOut, OutFun = TRSignal)
        # reversed signal is reemitted to focus an impulse on a particular point on the plate
    
        max_amplitude[k] = max(np.abs(result)) # save the maximum amplitude to compute the PSF
        energy[k] = np.trapz(np.abs(result[:, 0])**2, x = time_result) # integral of the squared signal
        
        #plt.plot(time_result, result, 'k')
        #plt.show()
        '''
        xF = np.fft.rfft(result)
        N = len(result)
        xF = xF[0 : N/2]
        freq = np.linspace(0, m.RATE/2, N/2)
        
        mean_freq[k] = np.trapz(abs(xF)*freq) / np.trapz(abs(xF))
        '''
        '''
        if ChannelsPlate[k] == 8:
            plt.plot(time_result, result, 'k')
            plt.xlabel("Time [s]")
            plt.ylabel("Counts")
            plt.title("Impulse focused on sensor " + str(ChannelsIn[0] + 1) + ", recorded on sensor " + str(ChannelsPlate[k] + 1))
            plt.rc('font', size = 18)
            plt.show()#block = False)

        '''
        
         
    Distance_vect = [0, 21.5, 8, 18.5, 25, 9.5] #[18.5, 11, 12, 0, 7.5, 16] # distance from sensor 12
    PSF = np.column_stack((np.array(Distance_vect).reshape(6,1), np.array(max_amplitude)))
    PSF = PSF[PSF[:,0].argsort()]
    
    Distance_cm = PSF[:, 0]
    PSF_values = 10*np.log10(PSF[:, 1] / max(PSF[:, 1]))
    Frequency_range = [freq0, freq1]
    
    filename = savefilename + "_PSF"
    data = {'Distance_cm':Distance_cm, 'PSF':PSF_values, 'ChannelsPlate':ChannelsPlate, 'Frequency_range':Frequency_range, 'max_amplitude':max_amplitude, 'energy':energy, 'nb_hauts_parleurs':nb_capteurs[cc]}#, 'mean_freq_impulse':mean_freq}
    savemat(filename, data)
    '''
    # Plot the Point Spread Function (PSF)
    plt.plot(Distance_cm, PSF_values, 'k')
    plt.xlabel("Distance [cm]")
    plt.ylabel("PSF [dB]")
    plt.title(savefilename)
    plt.rc('font', size = 18)
    plt.show()
    '''