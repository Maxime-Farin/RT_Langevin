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
centerx = 0
centery = 280
centerz = 385


# Save directory
savefilename = '../Donnees_Langevin/20180118_test_vibro'
if not os.path.exists(savefilename):
    os.makedirs(savefilename)

print('Open serial ports...\n')
# Initialization motor
motor = imcs('COM6')
# Initialization vibrometer
vibro = OVF5000('COM7')
# Initialization class motu
m = motu()


#try: 
# Move to plate center
print('Move to plate center...\n')
motor.move(centerx, centery, centerz)
time.sleep(3)

if vibro.level() < 400: # if focus is not good
    vibro.autofocus() # function to make focus on the plate

# Output Loudspeakers channels
ChannelsOut = list(range(32))#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Channels of the loudspeakers that will play a signal
#ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
ChannelVibro = [1] #[9] # vibrometer channel
ChannelVibro = [c - 1 for c in ChannelVibro]


freq0 = 1000 # minimum frequency of the chirp
freq1 = 9000 # maximum frequency of the chirp

print('Play chirp...\n')
# emits a chirp from each ChannelsOut successively and record it on ChannelIn
impulse = np.zeros([ceil(m.RATE*m.dureeImpulse), len(ChannelsOut)])
for k in range(len(ChannelsOut)):
    impulse[:, k] = m.ChirpRec(ChannelVibro, ChannelsOut[k], freq0 = freq0, freq1 = freq1)[:, 0] # reponse impulsionnelle
    time.sleep(0.5)

time_impulse = np.arange(0.0, m.dureeImpulse, 1.0 / float(m.RATE))
# signal is reversed temporally and normalized to 1
TRSignal = impulse[: : -1, :] / np.abs(impulse).max() 

# Successive positions to scan
Positions = [c*10 + 80 for c in list(range(51))]

time_result = np.arange(0.0, m.RECORD_SECONDS, 1.0 / float(m.RATE)) 
max_amplitude = np.zeros([len(Positions), 1], dtype = np.float)
energy = np.zeros([len(Positions), 1], dtype = np.float)

print('Focus impulse on plate...\n')
for pp in range(len(Positions)):
    # Move motor to measurement position
    y = Positions[pp]
    print('Change position y to ' + str(y))
    motor.move(centerx, y, centerz)
    time.sleep(5)
    
    # Send reversed impulse focused on initial position
    result = m.PlayAndRec(ChannelVibro, ChannelsOut, OutFun = TRSignal)

    # Save max_amplitude and energy of the vibration signal
    max_amplitude[pp] = max(np.abs(result)) # save the maximum amplitude to compute the PSF
    energy[pp] = np.trapz(np.abs(result[:, 0])**2, x = time_result) # integral of the squared signal

    # Save data
    savemat('%s/Position_y_%d'% (savefilename, pp), {'impulse':result, 'time': time_result, 'z':centerz, 'y':y, 'x':centerx})
    
    time.sleep(5)


pos = [c - centery for c in Positions]

plt.cla()
plt.plot(pos, max_amplitude, 'k', linewidth = 2)
plt.xlabel("Distance from focus position [mm]")
plt.ylabel("Amplitude of the recorded impulse [Counts]")
#plt.legend()
plt.rc('font', size = 18)
plt.show()   


PSF_values = 10*np.log10(max_amplitude / max(max_amplitude))

plt.cla()
plt.plot(pos, PSF_values, 'k', linewidth = 2)
plt.xlabel("Distance from focus position [mm]")
plt.ylabel("PSF [dB]")
plt.rc('font', size = 18)
plt.show()  


savemat(savefilename + '/All_positions_y', {'Positions': pos, 'max_amplitude': max_amplitude, 'energy':energy, 'PSF':PSF_values})
'''
    motor.ser.close()        
    vibro.ser.close()
    m.stream.stop_stream()
    m.stream.close()
    m.p.terminate()
    
except:
    motor.ser.close()
    vibro.ser.close()
    m.stream.stop_stream()
    m.stream.close()
    m.p.terminate()
'''