#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:54:21 2019

@author: jay

This is a NumPy implementation of the waveform filtering
"""

import numpy as np

class firFilterA:        
    def __c_impres(self):
        k  = np.arange(0,self.taps)
        impres = np.zeros(k.size)
        if (not self.fc==0):
            wn = 2*np.pi*self.fc/self.fs
        L=(self.taps-1)/2   
        if self.fc==0:
           impres=0.5/self.taps;
        else:
           np.divide(2*self.fc/self.fs*np.sin((k-L)*wn),(k-L)*wn, out=impres, where=k!=L)
           impres[np.where(k==L)]=2*self.fc/self.fs
        return impres
           
    # This function returns the filter coefficient b
    # of the FIR filter with the specs indicated in the parameters
    def __firCustom(self):
        impr = self.__c_impres()
        
        k  = np.arange(0,self.taps)
        # Initialize coef
        coef = np.zeros(k.size)
        
        if (self.window == "rect"):
            # rect
            coef = np.ones((self.taps,1))
        elif (self.window == "hann"):
            # hanning
            coef = 0.5-0.5*np.cos(2*np.pi*(k+1)/(self.taps+1))
        elif (self.window == "hamm"):
            # hamming
            coef = 25/46-21/46*np.cos(2*np.pi*(k+1)/(self.taps+1))
        elif (self.window == "bman"):
            # blackman
            coef = .42-.5*np.cos(2*np.pi*(k+1)/(self.taps+1))+0.08*np.cos(4*np.pi*(k+1)/(self.taps+1))
        elif (self.window == "bmanh"):
            #blackman-harris
            coef = .423-.498*np.cos(2*np.pi*(k+1)/(self.taps+1))+0.0792*np.cos(4*np.pi*(k+1)/(self.taps+1))
        else:
            coef = np.ones((self.taps,1))
        
        #normalize
        b = np.multiply(coef,impr)
        
        return b/np.sum(b)
    
    def __init__(self, taps, fc , fs, filtertype = "low", window = "hann"):
        self.taps = taps
        self.fc = fc
        self.fs = fs
        self.filtertype = filtertype
        self.window = window
        self.b = self.__firCustom()
    
    #apply filter
    # define a function that can convolve N waveforms with the same 2nd axis b 
    # this fxn expects an input_x w/c is a numpy 2d array MxN
    # where M is the # of samples and N is the number of waveforms
    # b should be a 1d numpy array
    def applyFIR(self, input_x):
        M,N = input_x.shape
        M2, = self.b.shape
        y = np.zeros((M+M2-1,N))
        for i in range(N):
            y[:,i] = np.convolve(input_x[:,i],self.b)
        return y
    
    def printCoeff(self):
        attrs = vars(self)
        print('\n'.join("+%s: %s" % item for item in attrs.items()))