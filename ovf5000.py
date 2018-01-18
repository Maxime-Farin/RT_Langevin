# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:59:18 2017

@author: LOA

Pour acquérir des données avec le vibromètre laser
"""
import serial
from time import sleep
class OVF5000:
    def __init__(self,com='COM3'):
        self.ser=serial.Serial()
        self.ser.port=com
        self.ser.timeout=1
        self.ser.baudrate=115200
        self.ser.parity=serial.PARITY_NONE
        self.ser.stopbits=serial.STOPBITS_ONE
        self.ser.bytesize=serial.EIGHTBITS
        self.ser._xonxoff=0 
        if self.ser.isOpen():
            self.ser.close()
        self.ser.open()
        self.ser.write('\n'.encode('utf-8'))
        self.ser.readline()
        self.ser.readline()
        print(self.wr('GetDevInfo,Controller,0,Name'))

    def __del__(self):
        self.ser.close()
        
    def level(self):
        rep=self.wr('ST?')
        #print(int(rep))
        return(int(rep))
        
    def autofocus(self):
        self.wr('Set,SensorHead,0,AutoFocusSpan,Full')
        self.wr('Set,SensorHead,0,AutoFocus,Search')
        print('Recherche focus en cours')
        for i in range(100):
            sleep(0.5)
            rep=self.wr('Get,SensorHead,0,AutoFocus')
            if rep!='Search':
                print('Focus OK')
                break
        
    def wr(self,string):
        string=string+'\n'
        self.ser.write(string.encode('utf-8'))
        rep=self.ser.readline().decode("utf-8")
        if len(rep)==0:
            print('ATTENTION PAS DE RETOUR :'+string[:-1])
        return(rep[:-1])
        
        
if __name__ == '__main__':
    vibro=OVF5000('COM5')
    #vibro=OVF5000('/dev/ttyUSB1')
    vibro.level()
    vibro.autofocus()