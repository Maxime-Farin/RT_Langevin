from motu import motu
from imcs8a import imcs
from ovf5000 import OVF5000
from scipy.io import savemat, loadmat
from scipy import signal
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
centery = 220
centerz = 300

ChannelsOut = list(range(32))
ChannelVibro = [1] #[9] # vibrometer channel
ChannelVibro = [c - 1 for c in ChannelVibro]

# Save directory
savefilename = '../../Documents/Donnees_locales_RT/20180123_Excitation_and_Scan_vibro_y_220_z_300'
if not os.path.exists(savefilename):
    os.makedirs(savefilename)

# load data and write it in matrix impulse
d = loadmat('../../Documents/Donnees_locales_RT/20180119_Full_Scan_vibro/Position_y_220_z_300_LP_17.mat')
impulse = np.zeros([len(d['impulse_mpersec'][0]), len(ChannelsOut)]) 
for cc in range(len(ChannelsOut)):
    d = loadmat('../../Documents/Donnees_locales_RT/20180119_Full_Scan_vibro/Position_y_%d_z_%d_LP_%d.mat'% (centery, centerz, ChannelsOut[cc] + 1))
    impulse[:, cc] = d['impulse_mpersec'][0]

# invert impulse matrix temporally
TRSignal = impulse[: : -1, :] / np.abs(impulse).max() 

# Scan positions
Positions_y = [c*5 + 60 for c in list(range(105))]
Positions_z = [c*5 + 180 for c in list(range(83))]

time_result = np.arange(0.0, m.RECORD_SECONDS, 1.0 / float(m.RATE)) 
#max_amplitude = np.zeros([len(Positions_y), len(Positions_z)], dtype = np.float)
#energy = np.zeros([len(Positions_y), len(Positions_z)], dtype = np.float)

print('Focus impulse on plate...\n')
for zz in range(len(Positions_z)):

    for yy in range(len(Positions_y)):

        # Move motor to measurement position
        y = Positions_y[yy]
        z = Positions_z[zz]
        
        print('Change position to (' + str(y) + ',' + str(z) + ')')
        motor.move(centerx, y, z)
        time.sleep(5)
        
        # Re-do focus if not good   
        if vibro.level() < 400: # if focus is not good
            vibro.autofocus() # function to make focus on the plate
        
        time.sleep(1)
        
        # Send reversed impulse focused on initial position
        result = m.PlayAndRec(ChannelVibro, ChannelsOut, OutFun = TRSignal)

        # Save max_amplitude and energy of the vibration signal
        #max_amplitude[yy, zz] = max(np.abs(result)) # save the maximum amplitude to compute the PSF
        #energy[yy, zz] = np.trapz(np.abs(result[:, 0])**2, x = time_result) # integral of the squared signal

        # Save data
        savemat('%s/Position_y_%d_z_%d'% (savefilename, y, z), {'impulse':result, 'time': time_result, 'z':z, 'y':y, 'x':centerx, 'y_excitation': centery, 'z_excitation': centerz})
        
        time.sleep(5)

print('Scan Completed')
    
'''
# Position of the excitation point
centerx = 0
centery = 370
centerz = 380

## Load data and plot
# Save directory
savefilename = '../20180122_Excitation_and_Scan_vibro'
if not os.path.exists(savefilename):
    os.makedirs(savefilename)
# Scan positions
Positions_y = [c*5 + 270 for c in list(range(41))]
Positions_z = [c*5 + 280 for c in list(range(41))]


d = loadmat('%s/Position_y_%d_z_%d'% (savefilename, 270, 280))
time_result = d['time'][0] 
max_amplitude = np.zeros([len(Positions_y), len(Positions_z)], dtype = np.float)
energy = np.zeros([len(Positions_y), len(Positions_z)], dtype = np.float)

print('Focus impulse on plate...\n')
for zz in range(len(Positions_z)):

    for yy in range(len(Positions_y)):

        # Move motor to measurement position
        y = Positions_y[yy]
        z = Positions_z[zz]
        
        d = loadmat('%s/Position_y_%d_z_%d'% (savefilename, y, z))
        result = d['impulse']*0.005
        # Save max_amplitude and energy of the vibration signal
        max_amplitude[yy, zz] = max(np.abs(result)) # save the maximum amplitude to compute the PSF
        energy[yy, zz] = np.trapz(np.abs(result[:,0])**2, x = time_result) # integral of the squared signal


PSF_values = 10*np.log10(max_amplitude / max_amplitude.max())

savemat('%s/All_data_ExcitationPosition_y_%d_z_%d'% (savefilename, centery, centerz), {'Positions_y':Positions_y, 'Positions_z': Positions_z, 'max_amplitude':max_amplitude, 'energy': energy, 'PSF':PSF_values})
print('Data Saved')    

'''