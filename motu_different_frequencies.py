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

FREQ0 = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000]
FREQ1 = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]

ChannelsOut = list(range(32))
ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
ChannelsIn = [9] #capteur piezo sur la plaque
ChannelsIn = [c - 1 for c in ChannelsIn]
    

for ff in range(len(FREQ0)):

    freq0 = FREQ0[ff]
    freq1 = FREQ1[ff]

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
    
    
    savefilename = "20180109_32capteurs_" + str(FREQ0[ff]) + "_" + str(FREQ1[ff]) + "Hz"
    ChannelsPlate = [9, 10, 11, 12, 13, 14]
    ChannelsPlate = [c - 1 for c in ChannelsPlate]

    max_amplitude = np.zeros([len(ChannelsPlate), 1], dtype = np.float)
    energy = np.zeros([len(ChannelsPlate), 1], dtype = np.float)
    
    for k in range(len(ChannelsPlate)):
        result = m.PlayAndRec(ChannelsPlate[k], ChannelsOut, OutFun = TRSignal)
        # reversed signal is reemitted to focus an impulse on a particular point on the plate
    
        max_amplitude[k] = max(np.abs(result)) # save the maximum amplitude to compute the PSF
        energy[k] = np.trapz(np.abs(result[:, 0])**2, x = time_result) # integral of the squared signal
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
    Frequency_range = [FREQ0[ff], FREQ1[ff]]
    
    filename = savefilename + "_PSF"
    data = {'Distance_cm':Distance_cm, 'PSF':PSF_values, 'ChannelsPlate':ChannelsPlate, 'Frequency_range':Frequency_range, 'max_amplitude':max_amplitude, 'energy':energy}
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