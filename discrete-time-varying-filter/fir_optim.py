# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt

fs=1000;
fc=50;
t_len = 7.5 #seconds
t = np.arange(0,t_len
              ,1/fs)
x = 1 + 2*np.cos(2*np.pi*fc*t)
plt.plot(t,x)

#expected Total power
Ptot_e = 1**2 + (2**2)/2

#get fourier transform
x_fft = fftpack.fft(x)
Ptot_pars = np.sum(np.square(abs(x_fft)/x_fft.shape[0]))

#next example: sum of multiple sinusoids
fc = np.asarray([4.7, 11, 23, 26.5, 35])
a = np.asarray([1, 5, 2, 3, 4])
x_mult = np.zeros(t.shape)
it = np.nditer(fc, flags=['f_index'])
while not it.finished:
    x_mult = x_mult + a[it.index]*np.cos(2*np.pi*fc[it.index]*t)
    it.iternext()
plt.plot(t, x_mult)

Pmulti_exp = np.sum(np.square(a)/2)
#get fft
x_mult_fft = fftpack.fft(x_mult)
P_parseval = np.sum(np.square(abs(x_mult_fft)/x_mult_fft.shape[0]))
