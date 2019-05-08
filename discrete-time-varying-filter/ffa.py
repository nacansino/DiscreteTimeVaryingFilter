#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:54:21 2019

@author: jay

This is a NumPy implementation of the waveform filtering
"""

import numpy as np


class FIRFilterA:
    def __init__(self, taps, f_c, f_s, filter_type="low", window="hann"):
        self.taps = taps
        self.f_c = f_c
        self.f_s = f_s
        self.filter_type = filter_type
        self.window = window
        self.b_coeff = self.__firCustom()

    def __c_impres(self):
        """Returns the time-domain representation of the frequency-response impulse"""
        k = np.arange(0, self.taps)
        impres = np.zeros(k.size)
        if not self.f_c == 0:
            w_n = 2 * np.pi * self.f_c / self.f_s
        l = (self.taps - 1) / 2
        if self.f_c == 0:
            impres = 0.5 / self.taps
        else:
            np.divide(2 * self.f_c / self.f_s * np.sin((k - l) * w_n),
                      (k - l) * w_n, out=impres, where=k != l)
            impres[np.where(k == l)] = 2 * self.f_c / self.f_s
        return impres

    def __firCustom(self):
        """This function returns the filter coefficient b
        of the FIR filter with the specs indicated in the parameters"""
        impr = self.__c_impres()

        k = np.arange(0, self.taps)
        # Initialize coef
        coef = np.zeros(k.size)
        valid_window = ["rect", "hann", "hamm", "bman", "bmanh"]
        #check window
        if self.window not in valid_window:
            raise ValueError('"%s" window not supported' % self.window)
            
        if self.window == "rect":
            # rect
            coef = np.ones((self.taps,))
        elif self.window == "hann":
            # hanning
            coef = 0.5 - 0.5 * np.cos(2 * np.pi * (k + 1) / (self.taps + 1))
        elif self.window == "hamm":
            # hamming
            coef = 25 / 46 - 21 / 46 * \
                np.cos(2 * np.pi * (k + 1) / (self.taps + 1))
        elif self.window == "bman":
            # blackman
            coef = .42 - .5 * np.cos(2 * np.pi * (k + 1) / (self.taps + 1)) + \
                0.08 * np.cos(4 * np.pi * (k + 1) / (self.taps + 1))
        elif self.window == "bmanh":
            # blackman-harris
            coef = .423 - .498 * np.cos(2 * np.pi * (k + 1) / (self.taps + 1)) + \
                0.0792 * np.cos(4 * np.pi * (k + 1) / (self.taps + 1))
        else:
            coef = np.ones((self.taps,))

        # normalize
        b_coeff = np.multiply(coef, impr)

        return b_coeff / np.sum(b_coeff)

    def apply_filter(self, input_x):
        """apply filter
         define a function that can convolve N waveforms with the same 2nd axis b
         this fxn expects an input_x w/c is a numpy 2d array MxN
         where M is the # of samples and N is the number of waveforms
         b should be a 1d numpy array"""
        
        input_x = np.asarray(input_x) #convert to numpy array first
        m_x, n_x = input_x.shape
        m_b, = self.b_coeff.shape
        y_out = np.zeros((m_x + m_b - 1, n_x))
        for i in range(n_x):
            y_out[:, i] = np.convolve(input_x[:, i], self.b_coeff)
        return y_out

    def print_param(self):
        """print the filter object's parameters"""
        attrs = vars(self)
        print('\n'.join("+%s: %s" % item for item in attrs.items()))
