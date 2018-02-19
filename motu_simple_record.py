from motu import motu
from scipy.io import savemat
from math import *
import numpy as np
import os
import time
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt


# Initialization class motu
m = motu() # define a class motu named m

freq0 = 1000
freq1 = 10000
duree = 1.0

ChannelsOut = [25] # Output channels
ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
ChannelsIn = [2] #microphone
ChannelsIn = [c - 1 for c in ChannelsIn]

# load data and write it in matrix impulse
d = loadmat('../../Documents/Donnees_locales_RT/20180119_Full_Scan_vibro/Position_y_220_z_300_LP_17.mat')
impulse = np.zeros([len(d['impulse_mpersec'][0]), len(ChannelsOut)], dtype = np.float)
for cc in range(len(ChannelsOut)):
    d = loadmat('../../Documents/Donnees_locales_RT/20180119_Full_Scan_vibro/Position_y_%d_z_%d_LP_%d.mat'% (220, 300, ChannelsOut[cc] + 1))
    impulse[:, cc] = d['impulse_mpersec'][0]

# invert impulse matrix temporally
TRSignal = impulse[: : -1, :] / np.abs(impulse).max() 

result = m.PlayAndRec(ChannelsIn, ChannelsOut, OutFun = TRSignal)
time_result = np.arange(0.0, m.RECORD_SECONDS, 1.0 / float(m.RATE))


# Plot of the impulse recorded on the plate after time reversal
plt.plot(time_result, result, 'k')
plt.xlabel("Time [s]")
plt.ylabel("Counts")
plt.rc('font', size = 18)
plt.show()

savemat('20180213_microphone_HP25_Gain54dB_HiGain_impulsion_excitation_plaque_2', {'impulse':result, 'time': time_result})
