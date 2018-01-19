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
centery = 320
centerz = 385


# Save directory
savefilename = '../Donnees_Langevin/20180119_Full_Scan_vibro'
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


# Output Loudspeakers channels
ChannelsOut = list(range(32))#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Channels of the loudspeakers that will play a signal
#ChannelsOut = [c - 1 for c in ChannelsOut] # must retain 1 because sensor number = python number +1
ChannelVibro = [1] #[9] # vibrometer channel
ChannelVibro = [c - 1 for c in ChannelVibro]

freq0 = 1000 # minimum frequency of the chirp
freq1 = 10000 # maximum frequency of the chirp

Positions_y = [c*50 + 370 for c in list(range(11))]
Positions_z = [c*40 + 300 for c in list(range(11))]
time_impulse = np.arange(0.0, m.dureeImpulse, 1.0 / float(m.RATE))

# Move to initial scan position
print('Move to plate center...\n')
motor.move(centerx, Positions_y[0], Positions_z[0])
time.sleep(5)

if vibro.level() < 400: # if focus is not good
    vibro.autofocus() # function to make focus on the plate


for zz in range(len(Positions_z)):

    for yy in range(len(Positions_y)):

        # Move motor to measurement position
        y = Positions_y[yy]
        z = Positions_z[zz]
        print('Change position y to (' + str(y) + ',' + str(z) + ')')
        motor.move(centerx, y, z)
        time.sleep(5)
        
        # emits a chirp from each ChannelsOut successively and record it on ChannelIn
        
        print('Play chirp...\n')
        for k in range(len(ChannelsOut)):
            impulse = np.zeros([ceil(m.RATE*m.dureeImpulse), 1])
            impulse = m.ChirpRec(ChannelVibro, ChannelsOut[k], freq0 = freq0, freq1 = freq1)[:, 0] # reponse impulsionnelle
            impulse = impulse*0.005; # convert voltage into m/s
            
            # Save recorded impulse at position (y,z)
            savemat('%s/Position_y_%d_z_%d_LP_%d'% (savefilename, y, z, ChannelsOut[k]+1), {'impulse_mpersec':impulse, 'time': time_impulse, 'z':z, 'y':y, 'x':centerx, 'Loudspeaker':ChannelsOut[k]+1, 'centery': centery, 'centerz': centerz})
            time.sleep(1)
                
        time.sleep(5)
    
    # Re-do focus at each different z   
    if vibro.level() < 400: # if focus is not good
        vibro.autofocus() # function to make focus on the plate
