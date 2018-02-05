from motu import motu
from scipy.io import savemat
from math import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt


m = motu() # define a class motu named m

freq0 = 1000
freq1 = 10000

ChannelsOut = [24] # Output channels
ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
ChannelsIn = [2] #capteur piezo sur la plaque
ChannelsIn = [c - 1 for c in ChannelsIn]

result = m.PlayAndRec(ChannelsIn, ChannelsOut, freq0 = freq0, freq1 = freq1)
time_result = np.arange(0.0, m.RECORD_SECONDS, 1.0 / float(m.RATE)) 

# Plot of the impulse recorded on the plate after time reversal
plt.plot(time_result, result, 'k')
plt.xlabel("Time [s]")
plt.ylabel("Counts")
plt.title("Impulse focused on plate piezo")
plt.rc('font', size = 18)
plt.show()

savemat('20180205_Test_son_Gain54dB', {'impulse':result, 'time': time_result})
