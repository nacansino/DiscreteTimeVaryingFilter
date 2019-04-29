#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:15:38 2019

@author: jay

Creation of Gaussian fitness function
"""

import numpy as np

tx=np.arange(0,1,0.01)
ty=np.arange(-0.5,0.5,0.00001)

sx=0.4
sy=0.3

x,y = np.meshgrid(tx,ty,sparse=True)
gauss=100*np.exp(-1*((x**2/(2*sx**2))+(y**2/(2*sy**2))))
plt.contourf(tx,ty,gauss)

#write tne fxn
def gaussff(x,y,sx,sy):
    return 100*np.exp(-1*((x**2/(2*sx**2))+(y**2/(2*sy**2))))