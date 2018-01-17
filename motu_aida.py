import pyaudio
import wave
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import matplotlib.pyplot as plt
from math import *
import time


class motu:
    def __init__(self):
        self.CHUNK = 1024*8
        self.FORMAT = pyaudio.paInt32
        self.CHANNELS = 16
        self.RATE = 96000*2
        self.dureeImpulse = 2.0
        self.RECORD_SECONDS = self.dureeImpulse+0.5
        t=np.arange(0.0,self.dureeImpulse,1.0 /float(self.RATE))
        self.chirp=signal.chirp(t,1500,t.max(),90000,'linear',90.0)
        self.chirp=self.chirp*2**31
        self.chirp=self.chirp.astype('int32')
        flagFound=False
        self.p = pyaudio.PyAudio()
        for iDevice in range(self.p.get_device_count()):
            device=self.p.get_device_info_by_index(iDevice)
            #if (device['maxInputChannels']==23 and device['hostApi']==2):
            #if (device['maxInputChannels']==24 and device['hostApi']==2 and  device['name'].find('MOTU')>=0):
            if (device['maxInputChannels']==24 and device['hostApi']==2 and  device['name'].find('OrionTB')>=0):
                print(device)
                flagFound=True
                break

        if not flagFound:
            for iDevice in range(self.p.get_device_count()):
                device=self.p.get_device_info_by_index(iDevice)
                if device['hostApi']==2:
                    print(device)
            print('device not found')
            quit()
        self.index=-1
        self.dataOut=np.zeros([self.CHUNK,self.CHANNELS],dtype=np.int32)
        self.dataOut0=np.zeros([self.CHUNK,self.CHANNELS],dtype=np.int32)
        self.data4=np.zeros([self.CHUNK,self.CHANNELS],dtype=np.int32)
        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        input=True,
                        output=True,
                        rate=self.RATE,
                        input_device_index = iDevice,
                        output_device_index = iDevice,
                        frames_per_buffer=self.CHUNK,
                        stream_callback=self.callback)
                        
        self.stream.start_stream()

    
    def callback(self,in_data, frame_count, time_info, status):
        #for i in range(0, int(len(result) // self.CHUNK)+1):
        if self.index==-1:
            return (self.dataOut0.tostring(), pyaudio.paContinue)
            
        self.data4[:,:] = np.fromstring(in_data, dtype=np.int32).reshape([self.CHUNK,self.CHANNELS])
        for k in range(len(self.ChannelsIn)):
            self.data3[:,k]=self.data4[:,self.ChannelsIn[k]]
        
        if (self.index*self.CHUNK)<self.result.shape[0]:
            if ((self.index+1)*self.CHUNK>self.result.shape[0]):
                endRIndex=self.result.shape[0]
                endLIndex=self.result.shape[0]-self.index*self.CHUNK
                self.result[self.index*self.CHUNK:endRIndex,:]=self.data3[:endLIndex,:]
            else:
                self.result[self.index*self.CHUNK:(self.index+1)*self.CHUNK,:]=self.data3[:,:]

               
        if (self.index*self.CHUNK)<len(self.chirp):
            if ((self.index+1)*self.CHUNK>len(self.chirp)):
                endRIndex=len(self.chirp)
                endLIndex=len(self.chirp)-self.index*self.CHUNK
                for k in range(len(self.ChannelsOut)):
                    self.dataOut[:endLIndex,self.ChannelsOut[k]]=self.chirp[self.index*self.CHUNK:endRIndex]*self.Amplitude[k]
                    self.dataOut[endLIndex:,self.ChannelsOut[k]]=0
            else:
                for k in range(len(self.ChannelsOut)):
                    self.dataOut[:,self.ChannelsOut[k]]=self.chirp[self.index*self.CHUNK:(self.index+1)*self.CHUNK]*self.Amplitude[k]
        else:
            self.dataOut[:,self.ChannelsOut]=0
        
        self.index=self.index+1
        if self.index==int(len(self.result) // self.CHUNK)+1:
            self.index=-1
            
        return (self.dataOut.tostring(), pyaudio.paContinue)

        
    def impulseRec(self,ChannelsIn,ChannelsOut,Amplitude=1):
        if Amplitude==1:
            Amplitude=[1]*len(ChannelsOut)
        
        if len(Amplitude)!=len(ChannelsOut):
            throw('ChannelsOut and Amplitude dims should be equal')
        
        self.Amplitude=Amplitude
        self.ChannelsIn=ChannelsIn
        self.ChannelsOut=ChannelsOut
        self.data3=np.zeros([self.CHUNK,len(ChannelsIn)],dtype=np.int32)
        self.result=np.zeros((int(self.RECORD_SECONDS*self.RATE),len(ChannelsIn)),dtype=np.int32)
        
        self.index=0
        while self.index!=-1:
            time.sleep(0.1)
            
        result=self.result.astype(dtype=np.float32)

        nb=2**(ceil(log(result.shape[0])/log(2)))
        #print(nb)
        #plt.plot(result)
        #plt.show()
        ftchirp=np.fft.rfft(self.chirp,nb)
        impulse=np.zeros((ceil(self.RATE*self.dureeImpulse),result.shape[1]))
        for k in range(len(ChannelsIn)):
            ftresult=np.fft.rfft(result[:,k],nb)
            corre=np.fft.irfft(ftresult*np.conjugate(ftchirp))
            impulse[:,k]=corre[:impulse.shape[0]]

        return(impulse)
        
    def __close__(self):
        self.stream.close()
        self.p.terminate()
        del p
        

        
if __name__== '__main__':
    m=motu()
    plt.cla()
    for i in range(1):
        impulse=m.impulseRec([2-1,1-1],[7-1])
    plt.plot(impulse[:,0])
    plt.show()