from motu import motu
from imcs8a import imcs
from ovf5000 import OVF5000
from scipy.io import savemat, loadmat
from math import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt


# Position of the center of the plate
centerx = 0
centery = 315
centerz = 320


# Save directory
savefilename = '../../Documents/Donnees_locales_RT/20180219_Linear_Scan_Parallel_Pipe'
if not os.path.exists(savefilename):
    os.makedirs(savefilename)

print('Open serial ports...\n')
# Initialization motor
motor = imcs('COM13')
# Initialization vibrometer
vibro = OVF5000('COM12')
# Initialization class motu
m = motu()


#try: 


# Output Loudspeakers channels
ChannelsOut = list(range(32))#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Channels of the loudspeakers that will play a signal
#ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
ChannelVibro = [1] #[9] # vibrometer channel
ChannelVibro = [c - 1 for c in ChannelVibro]

duree = 1
m.RATE = 48000*4;
freq0 = 1000 # minimum frequency of the chirp
freq1 = 10000 # maximum frequency of the chirp

Positions_y = [c*5 + 60 for c in list(range(103))]
Positions_z = [320]
#Positions_y = [520, 570]
#Positions_z = [380]
time_impulse = np.arange(0.0, duree, 1.0 / float(m.RATE))



for zz in range(len(Positions_z)):
    for yy in range(len(Positions_y)):

        # Move motor to measurement position
        y = Positions_y[yy]
        z = Positions_z[zz]
        print('Change position y to (' + str(y) + ',' + str(z) + ')')
        motor.move(centerx, y, z)
        time.sleep(5)
        
        if vibro.level() < 400: # if focus is not good
            vibro.autofocus() # function to make focus on the plate
            time.sleep(1)
            
        # emits a chirp from each ChannelsOut successively and record it on ChannelIn
        print('Play chirp...\n')
        for k in range(len(ChannelsOut)):
            impulse = np.zeros([ceil(m.RATE*duree), 1], dtype = np.float)
            impulse = m.ChirpRec(ChannelVibro, ChannelsOut[k], freq0 = freq0, freq1 = freq1, duree = duree)[:, 0] # reponse impulsionnelle
            
            # Save recorded impulse at position (y,z)
            savemat('%s/Position_y_%d_z_%d_LP_%d'% (savefilename, y, z, ChannelsOut[k]+1), {'impulse_counts':impulse, 'time_s': time_impulse, 'z':z, 'y':y, 'x':centerx, 'Loudspeaker':ChannelsOut[k]+1, 'centery': centery, 'centerz': centerz})
            time.sleep(1)
                
        time.sleep(5)
    
