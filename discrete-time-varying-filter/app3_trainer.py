#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 00:14:44 2019

@author: jay
This code demostrates the current filter optimization by GA
As of this writing, the GA code is running correctly but it isn't evolving as expected.
We will try to change the fitness function 
"""

import dtvf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wvf_plot(wvf):
    #plot a checkweigher waveform using matplotlib
    x = np.asarray(wvf)
    n = np.arange(0, x.shape[0])
    plt.plot(x)
    plt.xlabel('samples (n)')
    plt.ylabel('weight (grams)')
    plt.title('waveform filtering using time-varying filters')
    plt.legend()
    plt.show()    
    

#def main():
optimizer=dtvf.GA_OPtimizeDTVF(Ts = 0.001)
population = optimizer.init_population(size = 100)

#load sample data 1
df = pd.read_csv('data/sample.csv')
df = df.iloc[:,1:]


# Run the genetic algorithm here
num_generations = 20

new_population = population
for i in range(0,num_generations):
    fitness = optimizer.cal_pop_fitness(population = new_population, 
                                        meas_time = 99, 
                                        input_x = df, 
                                        realweight = 22.51)
    #select 5 fittest members (with the biggest probability)
    fittest_idx = np.argsort(fitness[:,1],)[:5]
    #for now, limit parents to 5
    new_population = optimizer.selection(new_population, fitness, 5)
    print("Gen ",i,
          ": top performer 3s,Xbar-x (",
          fitness[fittest_idx[0],1], ", ",
          fitness[fittest_idx[0],2], ")")