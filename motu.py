import pyaudio
import wave
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import matplotlib.pyplot as plt
from math import *
import time


class motu:
    """ motu class"""
    def __init__(self):
        ''' constructor of class motu '''
        self.CHUNK = 1024*8 # size of the buffer
        self.FORMAT = pyaudio.paInt32 # format of the data
        self.CHANNELS = 24 # number of channels
        self.RATE = 48000 # sampling rate
        self.dureeImpulse = 1.0 # duration of the emitted impulse
        self.RECORD_SECONDS = self.dureeImpulse + 0.5 # duration of the recording
        
        t = np.arange(0.0, self.dureeImpulse, 1.0 / float(self.RATE)) # definition of time vector t from 0 to 1 s
        
        # Chirp definition (chirp = emitted sound)
        self.chirp = signal.chirp(t, 100, t.max(), 3000) # cosine generator signal.chirp(time vector, f0 at t=0, t1, f1 at t1) 
        #self.chirp = self.chirp*2**30 # increase amplitude of the signal to be emitted to half the maximum possible amplitude in 32 bit
        #self.chirp = self.chirp.astype('int32') # format
        #flagFound = False # flagFound = False if the audio device is not found
        
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
                endRIndex =self.OutFun.shape[0]
                endLIndex =self.OutFun.shape[0] - self.index*self.CHUNK
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
        Play a chrip signals on ChannelsOut, 
        records them on ChannelsIn 
        and correlate 
        
        Arguments:
        ChannelsIn: input channels we want to use
        ChannelsOut: output channels we want to use
        '''
        
        result = self.PlayAndRec(ChannelsIn, ChannelsOut) # call PlayandRec function to send a chrip signal and store the recorded signal in result
        
        nb = 2**(ceil(log(result.shape[0]) / log(2))) # length of the fft (ceil(n) is the smallest integrer above n)
        ftchirp = np.fft.rfft(self.chirp, nb) # fft of the chirp signal (np = numpy)
        impulse = np.zeros((ceil(self.RATE*self.dureeImpulse), result.shape[1])) # initialization of the impulse matrix to reserve space
        for k in range(len(ChannelsIn)):
            ftresult = np.fft.rfft(result[:, k], nb) # ftt of the recorded signal
            corre = np.fft.irfft(ftresult*np.conjugate(ftchirp)) # do the correlation between the recorded signal and the emitted chirp on every channels
            impulse[:, k] = corre[:impulse.shape[0]] # store the correlation for each channel in the matrix impulse
        
        return(impulse) # return a vector of the correlation for each channel
        
        
    def PlayAndRec(self, ChannelsIn, ChannelsOut, OutFun=None, Amplitude = 1):
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
        
        if OutFun==None: # if no function is given in argument the default is a chirp
            self.OutFun = np.outer(self.chirp*2**30,np.ones(len(ChannelsOut)))
        else:
            self.OutFun = OutFun*2**30 # sets amplitude to half the possible maximum amplitude in 32 bit (max is 2**31)
            
            
        if Amplitude == 1:
            Amplitude = [1]*len(ChannelsOut)
        
        if len(Amplitude) != len(ChannelsOut):
            throw('ChannelsOut and Amplitude dims should be equal')
        
        self.Amplitude = Amplitude # sets the value of the output signal amplitude in the whole class
        self.ChannelsIn = ChannelsIn # sets the input channels in the whole class
        self.ChannelsOut = ChannelsOut # sets the output channels in the whole class
        self.data3 = np.zeros([self.CHUNK, len(ChannelsIn)], dtype = np.int32) # initialize output matrix to reserve space
        self.result = np.zeros((self.RECORD_SECONDS*self.RATE, len(ChannelsIn)), dtype = np.int32) # initialize input matrix
        
        # stop this function here until the recording is over (when self.index becomes -1 again)
        # so that the result vector is complete before returning it
        self.index = 0
        while self.index != -1:
            time.sleep(0.1)
        
        result = self.result.astype(dtype = np.float32) # complete recorded signal

        return(result) #return complete signal recorded on ChannelsIn
        
    def __close__(self):
        self.stream.close() # fermer le stream
        self.p.terminate()
        

        
if __name__== '__main__': # mettre ceci dans un if permet de lancer le script motu.py ou de l'utiliser comme library comme import motu.py sans que cette partie soit lue
    m = motu() # define a class motu named m
    plt.cla() # clear axis
    
    # if one want a chirp (for example to calibrate the response between sensor and receiver)
    impulse = m.ChirpRec(ChannelsIn = [8], ChannelsOut = [19]) 
    plt.plot(impulse)
    plt.show()
    
    
    
    '''
    # Output function definition
    t = np.arange(0, 1, 1.0 / float(m.RATE)) # definition of time vector t from 0 to 1 s
    MyFunction = np.column_stack((np.sin(t*2*np.pi*1000),np.sin(t*2*np.pi*400)))
 
    
    result = m.PlayAndRec(ChannelsIn = [8], ChannelsOut = [18], OutFun = MyFunction)
    plt.figure()
    plt.plot(result)
    plt.show()
    '''