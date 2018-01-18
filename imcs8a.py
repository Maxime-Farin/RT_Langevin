# -*- coding: utf-8 -*-
import serial
from time import sleep
class imcs:
    """Les commandes sont:
    mrot.start : permet de remettre le moteur à la position 0 (à faire au début d'une manip)
    mrot.init : permet de prendre l'actuelle position comme position de référence
    mrot.restart : permet de remettre le moteur à la position enregistée
    mrot.rot(N) : permet de faire pivoter le moteur de N degres"""
    nbAxis=3
    Imcs8aErrors={'1':'imcs8a: Error in numeric value provided',
            '2':'imcs8a: End switch error',
            '3':'imcs8a: Incorrect axis specification',
            '4':'imcs8a: No axis defined',
            '5':'imcs8a: Syntax error',
            '6':'imcs8a: End of memory',
            '7':'imcs8a: Incorrect number of parameters',
            '8':'imcs8a: Command to be stored incorrect',
            '9':'imcs8a: System error',
            'D':'imcs8a: Speed not permitted',
            'F':'imcs8a: User Stop',
            'G':'imcs8a: Invalid data field',
            'H':'imcs8a: Cover error',
            'R':'imcs8a: Reference error'}
    def __init__(self,com,init=False):
        self.ser=serial.Serial()
        self.ser.port=com
        self.ser.timeout=5
        self.ser.baudrate=19200
        self.ser.parity=serial.PARITY_NONE
        self.ser.stopbits=serial.STOPBITS_ONE
        self.ser.bytesize=serial.EIGHTBITS
        self.ser._xonxoff=0 
        if self.ser.isOpen():
            self.ser.close()
        self.ser.open()
        
        if init:
            if (self.nbAxis==3):
                self.write('@07');
                self.write('@0d6000,6000,6000')
                self.write('@0R7')
            elif (self.nbAxis==2):
                self.write('@03');
                self.write('@0d6000,6000')
                self.write('@0R3')            
            sleep(1)
            
    def write(self,chaine):
        chaine='{}\r\n'.format(chaine)
        chaine2=chaine.encode('utf-8')
      
        self.ser.write(chaine2)
        
        count=0
        while True:
            try:
                rep=self.ser.read(1)
                if rep==b'0':
                    break
                elif rep==b'':
                   count=count+1
                   print(count)
                elif rep==b'R':
                    if (self.nbAxis==3):
                        self.write('@07');
                        self.write('@0d6000,6000,6000')
                        self.write('@0R7')
                    elif (self.nbAxis==2):
                        self.write('@03');
                        self.write('@0d6000,6000')
                        self.write('@0R3')
                    sleep(1)
                    self.ser.write(chaine2)
                else:
                    raise  NameError(self.Imcs8aErrors[rep.decode('utf-8')])
            except serial.SerialTimeoutException:
                count=count+1
                print(count)
                if count>210/5:
                    raise serial.SerialTimeoutException

        
            
                    
    def move(self,x,y,z=0):
        pasparmm=160*2;
        xp=round(x*pasparmm);
        yp=round(y*pasparmm);
        zp=round(z*pasparmm);
        v=5000;
        if self.nbAxis==3:
            chaine='@0M{},{},{},{},{},{},0,21'.format(xp,v,yp,v,zp,v);
        elif self.nbAxis==2:
            chaine='@0M{},{},{},{},0,21'.format(xp,v,yp,v);
        self.write(chaine)


if __name__== '__main__':
    a = imcs('COM4')
    a.move(0, 340, 385)