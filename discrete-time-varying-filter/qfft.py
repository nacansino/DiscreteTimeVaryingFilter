# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:42:16 2019

@author: a1216021
"""

import numpy as np
from scipy.fftpack import fft

def qFFT(input_data, fs = 1000):
    x = np.asarray(input_data)
    
    n = x.shape[0]
    
    # apply fft
    yf = np.zeros((x.shape[0]//2, x.shape[1]))
    for i in range(x.shape[1]):
        yf[:,i] = 4.0/n * np.abs(fft(x[:,i])[0:n//2])        
    yf[0,:] = np.zeros((1,x.shape[1]))
    
    # create f-axis
    ff = np.linspace(0, fs/2-1, n//2)
    return (yf, ff)
