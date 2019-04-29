#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 00:14:44 2019

@author: jay
This code demostrates the current filter optimization by GA

Apr29 00:14
As of this writing, the GA code is running correctly but it isn't evolving as expected.
We will try to change the fitness function 

Apr29 14:47
Added GA_OptimizeDTVF.optimize.mutation()
This seems to add a certain degree of freedom to the optimizer.

Apr29 16:57
Created Gaussian fitness function
The GFF seems to work but it depends upon the initial population.
If the fittest member is in a false direction, there is a tendency to fall to a false minimum
"""

import dtvf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb #breakpoint generator

def wvf_plot(wvf):
    #plot a checkweigher waveform using matplotlib
    x = np.asarray(wvf)
    plt.plot(x)
    plt.xlabel('samples (n)')
    plt.ylabel('weight (grams)')
    plt.title('waveform filtering using time-varying filters')
    plt.legend()
    plt.show()    
    

#def main():
optimizer=dtvf.GA_OPtimizeDTVF(Ts = 0.001)
population = optimizer.init_population(size = 200)

#load sample data 1
df = pd.read_csv('data/sample.csv')
df = df.iloc[:,1:]


# Run the genetic algorithm here
num_generations = 20

#record fittest member for each generation

fittest_per_generation=[]
new_population = population
for i in range(0,num_generations):
    fitness = optimizer.cal_pop_fitness(population = new_population, 
                                        meas_time = 99, 
                                        input_x = df, 
                                        realweight = 22.51)
    #select 5 fittest members (with the biggest probability)
    fittest_idx = np.argsort(fitness[:,1],)[:5]
    
    print("Gen ",i,
          ": top performer 3s,Xbar-x (",
          fitness[fittest_idx[0],1], ", ",
          fitness[fittest_idx[0],2], ")")
    #print params for the fittest
    bestmember=[print(keys,values) for keys,values in new_population[fittest_idx[0]].items()]
    print("\n")
    fittest_per_generation.append(new_population[fittest_idx[0]])
    #pdb.set_trace()
    
    
    #generate new population
    #for now, limit parents to 5
    pop_selection = optimizer.selection(population = new_population, 
                                        fitness = fitness,
                                        num_parents=5)
    #add 20% mutation
    new_population = optimizer.mutation(population = pop_selection,
                                        rate = 0.2)