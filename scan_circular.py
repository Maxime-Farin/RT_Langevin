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
centerz = 230
centery = 325
centerx = 10

R = 153

# Save directory
directory = 'circ_with_d12_r153'
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialization motor
motor = imcs('COM4', False)
# Initialization vibrometer
vibro = OVF5000('COM5')
# Initialization class motu
m = motu()

nbangles=120

for i in range(nbangles):
    angle=2*3.14159/(nbangles)*i
    print(angle)
    y=centery+R*cos(angle)
    z=centerz+R*sin(angle)
    
    # Move to point (y,z) on the plane
    motor.move(z, y, centerx)
    time.sleep(5)
    
    if vibro.level() < 400: # if focus is not good
        vibro.autofocus() # function to make focus on the plate
    
    # Send a chirp
    impulse = m.impulseRec([2-1,1-1], [7-1])

    if (i == 0): # create empty vector of data
        imp = np.zeros((impulse.shape[0],impulse.shape[1],nbangles))

    # Collect data in vector imp
    imp[:,:,i] = impulse[:,:]

    # Save data
    savemat('%s/data_%d'% (directory,i+1),{'impulse':imp,'angle':angle,'z':z,'y':y,'x':centerx,'R':R})


plt.cla()
plt.plot(imp[:,0:])
plt.show()  
    