from motu import motu
from imcs8a import imcs
from ovf5000 import OVF5000
from scipy.io import savemat
from math import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt


# Position of the center of the plate
centerz = 385
centery = 340
centerx = 0


# Save directory
savefilename = '../Donnees_Langevin/20180118_test_vibro'
if not os.path.exists(savefilename):
    os.makedirs(savefilename)

# Initialization motor
motor = imcs('COM4', False)
# Initialization vibrometer
vibro = OVF5000('COM5')
# Initialization class motu
m = motu()


y = centery
z = centerz

# Move to plate center
motor.move(z, y, centerx)
time.sleep(5)

if vibro.level() < 400: # if focus is not good
    vibro.autofocus() # function to make focus on the plate

# Output Loudspeakers channels
ChannelsOut = [1, 2, 3, 4, 5, 6] # Channels of the loudspeakers that will play a signal
ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
ChannelVibro = [1] #[9] # vibrometer channel
ChannelVibro = [c - 1 for c in ChannelVibro]


freq0 = 1000 # minimum frequency of the chirp
freq1 = 9000 # maximum frequency of the chirp

# emits a chirp from each ChannelsOut successively and record it on ChannelIn
impulse = np.zeros([ceil(m.RATE*m.dureeImpulse), len(ChannelsOut)])
for k in range(len(ChannelsOut)):
    impulse[:, k] = m.ChirpRec(ChannelVibro, ChannelsOut[k], freq0 = freq0, freq1 = freq1)[:, 0] # reponse impulsionnelle
    time.sleep(0.5)

time_impulse = np.arange(0.0, m.dureeImpulse, 1.0 / float(m.RATE))
# signal is reversed temporally and normalized to 1
TRSignal = impulse[: : -1, :] / np.abs(impulse).max() 

# Successive positions to scan
Positions = centery + [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 

time_result = np.arange(0.0, m.RECORD_SECONDS, 1.0 / float(m.RATE)) 
max_amplitude = np.zeros([len(Positions), 1], dtype = np.float)
energy = np.zeros([len(Positions), 1], dtype = np.float)

for k in range(len(Positions)):
    # Move motor to measurement position
    y = Positions[k]
    motor.move(z, y, centerx)
    time.sleep(1)
    
    # Send reversed impulse focused on initial position
    result = m.PlayAndRec(ChannelVibro, ChannelsOut, OutFun = TRSignal)

    # Save max_amplitude and energy of the vibration signal
    max_amplitude[k] = max(np.abs(result)) # save the maximum amplitude to compute the PSF
    energy[k] = np.trapz(np.abs(result[:, 0])**2, x = time_result) # integral of the squared signal

    # Save data
    savemat('%s/Position_y_%d'% (savefilename, k),{'impulse':result, 'time': time_result, 'z':z, 'y':y, 'x':centerx, 'centery':centery, 'centerz':centerz})


plt.cla()
plt.plot(Positions - centery, max_amplitude, 'k', linewidth = 2)
plt.xlabel("Distance from focus position [cm]")
plt.ylabel("Amplitude of the recorded impulse [Counts]")
#plt.legend()
plt.rc('font', size = 18)
plt.show()   
plt.show()  
    