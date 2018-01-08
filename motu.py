#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pyaudio
import wave
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import matplotlib.pyplot as plt
from math import *
import time
import pickle


class motu:
    """ motu class"""
    def __init__(self):
        ''' constructor of class motu '''
        self.CHUNK = 1024*8 # size of the buffer
        self.FORMAT = pyaudio.paInt32 # format of the data
        self.CHANNELS = 24 # number of channels
        self.RATE = 48000*2 # sampling rate
        self.dureeImpulse = 1.0 # duration of the emitted impulse
        self.RECORD_SECONDS = self.dureeImpulse + 0.5 # duration of the recording

        t = np.arange(0.0, self.dureeImpulse, 1.0 / float(self.RATE)) # definition of time vector t from 0 to 1 s

        # Chirp definition (chirp = emitted sound)
        self.chirp = signal.chirp(t, 100, t.max(), 15000) # cosine generator signal.chirp(time vector, f0 at t=0, t1, f1 at t1)
        self.chirp  = self.chirp*np.hanning(t.shape[0]) # hanning window to taper signal
        #self.chirp = self.chirp*2**30 # increase amplitude of the signal to be emitted to half the maximum possible amplitude in 32 bit
        #self.chirp = self.chirp.astype('int32') # format
        flagFound = False # flagFound = False if the audio device is not found

        self.p = pyaudio.PyAudio() # Instantiate PyAudio (set up the portaudio system)

        # loop to look for the right audio device
        for iDevice in range(self.p.get_device_count()): # for each audio device
            device = self.p.get_device_info_by_index(iDevice)
            #if (device['maxInputChannels']==23 and device['hostApi']==2):
            if (device['maxInputChannels'] == self.CHANNELS and device['maxOutputChannels'] == self.CHANNELS and device['hostApi'] == 2):
            # if number sensors = self.CHANNEL (24) and sensors are I/O and have the right protocol ASIO
                flagFound = True # the audio device has been found
                break

        if not flagFound: # if the audio device has not been found
            for iDevice in range(self.p.get_device_count()):
                device = self.p.get_device_info_by_index(iDevice)
                if device['hostApi'] == 2: # look if one device have the right protocol and print its info
                    print(device)
            print('device not found')
            quit()

        self.index = -1
        self.dataOut = np.zeros([self.CHUNK, self.CHANNELS], dtype = np.int32) # define the data matrix to attribute some space for these matrix
        self.dataOut0 = np.zeros([self.CHUNK, self.CHANNELS], dtype = np.int32)
        self.data4 = np.zeros([self.CHUNK, self.CHANNELS], dtype = np.int32)

        # stream initialization
        self.stream = self.p.open(format = self.FORMAT,
                        channels = self.CHANNELS,
                        input = True,
                        output = True,
                        rate = self.RATE,
                        input_device_index = iDevice,
                        output_device_index = iDevice,
                        frames_per_buffer = self.CHUNK,
                        stream_callback = self.callback) # pointer to callback function

        self.stream.start_stream() # start the stream


    def callback(self, in_data, frame_count, time_info, status):
        '''
        callback is running when PyAudio needs new audio data to play (when the buffer is full)
        Returns the output vector: a chunk of length self.CHUNK of the signal over all channels
        '''
        #for i in range(0, int(len(result) // self.CHUNK)+1):
        if self.index == -1:
            return (self.dataOut0.tostring(), pyaudio.paContinue) # return 0 signal in output

        # mise en forme des vecteurs de donnees
        self.data4[:, :] = np.fromstring(in_data, dtype = np.int32).reshape([self.CHUNK, self.CHANNELS])# input data in_data are written in matrix data4
        for k in range(len(self.ChannelsIn)): # size depends on the number of channels we choose
            self.data3[:, k] = self.data4[:, self.ChannelsIn[k]] # self.data3 contains the input data

        # write the input data in self.result chunk by chunk
        if ((self.index + 1)*self.CHUNK > self.result.shape[0]): # if chirp length is longer than recording length (not likely)
            endRIndex = self.result.shape[0]
            endLIndex = self.result.shape[0] - self.index*self.CHUNK
            self.result[self.index*self.CHUNK : endRIndex, :] = self.data3[: endLIndex, :]
        else: # what to input in general
            self.result[self.index*self.CHUNK : (self.index + 1)*self.CHUNK, :] = self.data3[:, :]


        # write the output data in self.dataOut chunk by chunk
        if (self.index*self.CHUNK) < self.OutFun.shape[0]: # outputs the chirp signal in successive chunks
            if (self.index + 1)*self.CHUNK > self.OutFun.shape[0]: # if the last chunk index is larger than the maximum chirp index
                endRIndex = self.OutFun.shape[0]
                endLIndex = self.OutFun.shape[0] - self.index*self.CHUNK
                for k in range(len(self.ChannelsOut)): # for all output channels
                    # write chirp data in output channels
                    self.dataOut[: endLIndex, self.ChannelsOut[k]] = self.OutFun[self.index*self.CHUNK : endRIndex, k]*self.Amplitude[k]
                    self.dataOut[endLIndex :, self.ChannelsOut[k]] = 0
            else: # what to output in general
                for k in range(len(self.ChannelsOut)):
                    self.dataOut[:, self.ChannelsOut[k]] = self.OutFun[self.index*self.CHUNK : (self.index + 1)*self.CHUNK, k]*self.Amplitude[k]
        else:  # if we reached the end of the chirp, outputs 0 signal
            self.dataOut[:, self.ChannelsOut] = 0

        # increase index of the chunk
        self.index = self.index + 1
        if self.index == int(len(self.result) // self.CHUNK) + 1: # if the recording is over...
            self.index = -1 # ...we reset the index to -1

        return (self.dataOut.tostring(), pyaudio.paContinue) # return the output data to be emitted


    def ChirpRec(self, ChannelsIn, ChannelsOut):
        '''
        Play a chrip signals c(t) on ChannelsOut,
        records them on ChannelsIn (r(t))
        and correlate r(t) * c(-t) to return the impulse response h(t) between source and sensor

        Arguments:
        ChannelsIn: input channels we want to use
        ChannelsOut: output channels we want to use
        '''

        result = self.PlayAndRec(ChannelsIn, ChannelsOut) # call PlayandRec function to send a chrip signal and store the recorded signal in result
        # r(t) = h(t) * c(t) (h(t): impulsional response between source and sensor)

        nb = 2**(ceil(log(result.shape[0]) / log(2))) # length of the fft (ceil(n) is the smallest integrer above n)
        ftchirp = np.fft.rfft(self.chirp, nb) # fft(c(t)) fft of the chirp signal (np = numpy)
        impulse = np.zeros([ceil(self.RATE*self.dureeImpulse), result.shape[1]], dtype = np.int32) # initialization of the impulse matrix to reserve space
        for k in range(len(ChannelsIn)):
            ftresult = np.fft.rfft(result[:, k], nb) # ftt(r(t)) : fft of the recorded signal
            corre = np.fft.irfft(ftresult*np.conjugate(ftchirp)) # r(t) * c(-t) = h(t) * c(t) * c(-t)
            # do the correlation between the recorded signal and the conjugate of the emitted chirp on every channels
            impulse[:, k] = corre[:impulse.shape[0]] # store the correlation for each channel in impulse

        return(impulse) # return a vector of the impulsional response h(t) between each channel and the emitor for each channel


    def PlayAndRec(self, ChannelsIn, ChannelsOut, OutFun = None, Amplitude = 1):
        '''
        Play function OutFun on ChannelsOut,
        Records it on ChannelsIn and
        Return the recorded signals

        Arguments:
        ChannelsIn: input channels we want to use
        ChannelsOut: output channels we want to use
        OutFun: Function to play (default = chirp)
        Amplitude: Amplitude of the output signal (default = 1)
        '''

        if isinstance(ChannelsOut, int): # if ChannelsOut is only 1 integer, convert into list to compute len
            ChannelsOut = [ChannelsOut]
        if isinstance(ChannelsIn, int): # if ChannelsIn is only 1 integer, convert into list to compute len
            ChannelsIn = [ChannelsIn]    

        if OutFun is None: # if no function is given in argument the default is a chirp
            self.OutFun = np.outer(self.chirp*2**30,np.ones(len(ChannelsOut)))
        else:
            self.OutFun = OutFun*2**30 # sets amplitude to half the possible maximum amplitude in 32 bit (max is 2**31)

        #self.OutFun = self.OutFun/5

        if Amplitude == 1:
            Amplitude = [1]*len(ChannelsOut)

        if len(Amplitude) != len(ChannelsOut):
            throw('ChannelsOut and Amplitude dims should be equal')

        self.Amplitude = Amplitude # sets the value of the output signal amplitude in the whole class
        self.ChannelsIn = ChannelsIn # sets the input channels in the whole class
        self.ChannelsOut = ChannelsOut # sets the output channels in the whole class
        self.data3 = np.zeros([self.CHUNK, len(ChannelsIn)], dtype = np.int32) # initialize matrix to reserve space
        self.result = np.zeros([int(self.RECORD_SECONDS*self.RATE), len(ChannelsIn)], dtype = np.int32) # initialize input matrix

        # stop this function here until the recording is over (when self.index becomes -1 again)
        # so that the result vector is complete before returning it
        self.index = 0
        while self.index != -1:
            time.sleep(0.1)

        result = self.result.astype(dtype = np.float32) # complete recorded signal

        return(result) #return complete signal recorded on ChannelsIn


    def __close__(self):
        '''
        Close the stream
        '''
        self.stream.close()
        self.p.terminate()



if __name__== '__main__': # mettre ceci dans un if permet de lancer le script motu.py ou de l'utiliser comme library comme import motu.py sans que cette partie soit lue
    m = motu() # define a class motu named m
    plt.cla() # clear axis

    ChannelsOut = [1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23, 24]
    ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
    ChannelsIn = [12] #capteur piezo sur la plaque
    ChannelsIn = [c - 1 for c in ChannelsIn]

    # emits a chirp from each ChannelsOut successively and record it on ChannelIn
    impulse = np.zeros([ceil(m.RATE*m.dureeImpulse), len(ChannelsOut)], dtype = np.int32)
    for k in range(len(ChannelsOut)):
        impulse[:, k] = m.ChirpRec(ChannelsIn, ChannelsOut[k])[:, 0] # reponse impulsionnelle
        time.sleep(0.5)
    # if one want a chirp (for example to calibrate the response between source and receiver)

    time_impulse = np.arange(0.0, m.dureeImpulse, 1.0 / float(m.RATE))

    '''
    plt.plot(time_impulse, impulse)
    plt.xlabel("Time [s]")
    plt.ylabel("Counts")
    plt.show(block = False)
    '''

    # use the fact that the sound path is reversible h(-t) = h(t)
    TRSignal = impulse[: : -1, :] / np.abs(impulse).max() # signal is reserved temporally and normalized to 1
    time_result = np.arange(0.0, m.RECORD_SECONDS, 1.0 / float(m.RATE)) 
    
    
    savefilename = "20180801_Test_sensor"
    ChannelsPlate = [9, 10, 11, 12, 13, 14]
    ChannelsPlate = [c - 1 for c in ChannelsPlate]

    max_result = np.zeros([len(ChannelsPlate), 1], dtype = np.float)
    
    for k in range(len(ChannelsPlate)):
        result = m.PlayAndRec(ChannelsPlate[k], ChannelsOut, OutFun = TRSignal)
        # reversed signal is reemitted to focus an impulse on a particular point on the plate

    
        max_result[k] = max(result)


        plt.plot(time_result, result, 'k')
        plt.xlabel("Time [s]")
        plt.ylabel("Counts")
        plt.title("Impulse focused on sensor 12, recorded on sensor " + str(ChannelsPlate[k] + 1))
        plt.rc('font', size = 18)
        plt.show()#block = False)

    
        # write in filename
        filename = savefilename + "_" + str(ChannelsOut[2] + 1)

        with open(filename, 'wb') as fichier:
            p = pickle.Pickler(fichier)
            p.dump(time_impulse)
        with open(filename, 'ab') as fichier:
            p = pickle.Pickler(fichier)
            p.dump(impulse)
            p.dump(time_result)
            p.dump(result)
            p.dump(max_result[k])
            
            
    Distance_vect = [18.5, 11, 12, 0, 7.5, 16]
    PSF = np.column_stack((np.array(Distance_vect).reshape(6,1), np.array(max_result)))
    PSF = PSF[PSF[:,0].argsort()]
    
    Distance_cm = PSF[:, 0]
    PSF_values = np.log10(PSF[:, 1]) / np.log10(max(PSF[:, 1]))
    
    # Plot the Point Spread Function (PSF)
    plt.plot(Distance_cm, PSF_values, 'k')
    plt.xlabel("Distance [cm]")
    plt.ylabel("PSF")
    plt.title(savefilename)
    plt.rc('font', size = 18)
    plt.show()