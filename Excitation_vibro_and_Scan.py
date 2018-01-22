from motu import motu
from imcs8a import imcs
from ovf5000 import OVF5000
from scipy.io import savemat, loadmat
from math import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt



print('Open serial ports...\n')
# Initialization motor
motor = imcs('COM6')
# Initialization vibrometer
vibro = OVF5000('COM7')
# Initialization class motu
m = motu()



# Position of the excitation point
centerx = 0
centery = 370
centerz = 380

ChannelsOut = list(range(32))
ChannelVibro = [1] #[9] # vibrometer channel
ChannelVibro = [c - 1 for c in ChannelVibro]

# Save directory
savefilename = '../20180122_Excitation_and_Scan_vibro'
if not os.path.exists(savefilename):
    os.makedirs(savefilename)

# load data and write it in matrix impulse
d = loadmat('../20180119_Full_Scan_vibro/Position_y_370_z_380_LP_17.mat')
impulse = np.zeros([len(d['impulse_mpersec'][0]), len(ChannelsOut)]) 
for cc in range(len(ChannelsOut)):
    d = loadmat('../20180119_Full_Scan_vibro/Position_y_%d_z_%d_LP_%d'% (centery, centerz, ChannelsOut[cc] + 1))
    impulse[:, cc] = d['impulse_mpersec'][0]

# invert impulse matrix temporally
TRSignal = impulse[: : -1, :] / np.abs(impulse).max() 

# Scan positions
Positions_y = [c*5 + 270 for c in list(range(41))]
Positions_z = [c*5 + 280 for c in list(range(41))]

# Move to initial scan position
print('Move to first position...\n')
motor.move(centerx, Positions_y[0], Positions_z[0])
time.sleep(5)

if vibro.level() < 400: # if focus is not good
    vibro.autofocus() # function to make focus on the plate


time_result = np.arange(0.0, m.RECORD_SECONDS, 1.0 / float(m.RATE)) 
max_amplitude = np.zeros([len(Positions_y), len(Positions_z)], dtype = np.float)
energy = np.zeros([len(Positions_y), len(Positions_z)], dtype = np.float)

print('Focus impulse on plate...\n')
for zz in range(len(Positions_z)):

    for yy in range(len(Positions_y)):

        # Move motor to measurement position
        y = Positions_y[yy]
        z = Positions_z[zz]
        
        print('Change position y to (' + str(y) + ',' + str(z) + ')')
        motor.move(centerx, y, z)
        time.sleep(5)
        
        # Re-do focus if not good   
        if vibro.level() < 400: # if focus is not good
            vibro.autofocus() # function to make focus on the plate
        
        time.sleep(1)
        
        # Send reversed impulse focused on initial position
        result = m.PlayAndRec(ChannelVibro, ChannelsOut, OutFun = TRSignal)

        # Save max_amplitude and energy of the vibration signal
        max_amplitude[yy, zz] = max(np.abs(result)) # save the maximum amplitude to compute the PSF
        energy[yy, zz] = np.trapz(np.abs(result[:, 0])**2, x = time_result) # integral of the squared signal

        # Save data
        savemat('%s/Position_y_%d_z_%d'% (savefilename, y, z), {'impulse':result, 'time': time_result, 'z':z, 'y':y, 'x':centerx, 'y_excitation': centery, 'z_excitation': centerz})
        
        time.sleep(5)

        
PSF_values = 10*np.log10(max_amplitude / max(max_amplitude))

savemat('%s/All_data_ExcitationPosition_y_%d_z_%d'% (savefilename, centery, centerz), {'Positions_y':Positions_y, 'Positions_z': Positions_z, 'max_amplitude':max_amplitude, 'energy': energy, 'PSF':PSF_values})
print('Data Saved')    
