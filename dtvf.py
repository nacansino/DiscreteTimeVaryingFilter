#Discrete-Time Variable Filter implementation
#This code is based on the filtering algorithm on the ff:
#Original paper: Dynamic mass measurement using a discrete time-variant filter
#Author: Przemyslaw Pietrzak
#https://ieeexplore.ieee.org/document/5661899

import math
import numpy as np

DEBUG_MODE = False;

if DEBUG_MODE:
    print("Welcome to DTVF : Debug Mode")

class DiscTimeVarFilt:
    def apply_filter(self, x, ys = 0, xs = 0):
        x=np.asarray(x)
        N=x.size
        y=np.zeros((N,1))
        wc = self.w_inf + (self.w_o-self.w_inf)*self.alpha**(np.arange(0,N)/self.N_alpha)
        lmb = (-1)*((wc-2*self.xi/self.Ts)/(wc+2*self.xi/self.Ts))
        lmb_s = (1-lmb)/2

        #compute filtered value
        y[0] = lmb[0]*ys+lmb_s[0]*(x[0] + xs)
        for n in np.arange(1,N):
            #compute wc and lambda
            y[n] = lmb[n]*y[n-1]+lmb_s[n]*(x[0] + x[n-1])
        return y

    def __init__(self, Ts, f_o = 200, f_inf = 0.01, k = 2, N_alpha = 150, alpha = math.e):
        self.alpha = alpha
        self.N_alpha = N_alpha
        self.k = k
        self.w_o = f_o * 2 * math.pi
        self.w_inf = f_inf * 2 * math.pi
        self.Ts = Ts
        self.xi = math.sqrt(2**(k) - 1)
    #def set_params(self):


    #def apply_filter:

    #def apply_filter:
