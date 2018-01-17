from motu import motu
from imcs8a import imcs
from ovf5000 import OVF5000
from scipy.io import savemat
from math import *
import numpy as np
import os
import time
import matplotlib.pyplot as plt



centerz=230
centery=325
centerx=10

R=153

directory='circ_with_d12_r153'
if not os.path.exists(directory):
    os.makedirs(directory)

mot=imcs('COM3',False)
vibro=OVF5000('COM4')
m=motu()

nbangles=120

for i in range(nbangles):
    angle=2*3.14159/(nbangles)*i
    print(angle)
    y=centery+R*cos(angle)
    z=centerz+R*sin(angle)
    mot.move(z,y,centerx)
    time.sleep(5)
    if vibro.level()<400:
        vibro.autofocus()
    impulse=m.impulseRec([2-1,1-1],[7-1])
    if (i==0):
        imp=np.zeros((impulse.shape[0],impulse.shape[1],nbangles))
    imp[:,:,i]=impulse[:,:]
    savemat('%s/data_%d'% (directory,i+1),{'impulse':imp,'angle':angle,'z':z,'y':y,'x':centerx,'R':R})


plt.cla()
plt.plot(imp[:,0:])
plt.show()  
    