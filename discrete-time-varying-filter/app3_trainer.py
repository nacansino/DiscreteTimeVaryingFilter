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
#import pdb #breakpoint generator

def wvf_plot(wvf):
    #plot a checkweigher waveform using matplotlib
    x = np.asarray(wvf)
    plt.plot(x)
    plt.xlabel('samples (n)')
    plt.ylabel('weight (grams)')
    plt.title('waveform filtering using time-varying filters')
    plt.legend()
    plt.show()    
    

def evolution(init_population, input_x, optimizer, num_generations, fittest_threshold, size, mutation_rate=0.2):
    fittest_per_generation=[]
    new_population = init_population
    for i in range(0,num_generations):
        fitness = optimizer.cal_pop_fitness(population = new_population, 
                                            meas_time = 99, 
                                            input_x = input_x, 
                                            realweight = 22.51)
        #select 5 fittest members (with the biggest probability)
        fittest_idx = np.argsort(fitness[:,1],)[:5]
        
        print("Gen ",i+1,
              ": top performer 3s,Xbar-x (",
              fitness[fittest_idx[0],1], ", ",
              fitness[fittest_idx[0],2], ")")
        #print params for the fittest
        for keys,values in new_population[fittest_idx[0]].items():
            print(keys,values)
        print("\n")
        fittest_per_generation.append((fitness[fittest_idx[0],1],fitness[fittest_idx[0],2],new_population[fittest_idx[0]]))
        
        # Check fittest member
        # If fittest member has a fitness value of less than a certain defined threshold,
        # restart the whole evolution process
        
        if (fitness[fittest_idx[0],0] < fittest_threshold):
            #restart evolution process
            print("Encountered max(fitness)=",fitness[fittest_idx[0],0],"<",fittest_threshold,". Restarting evolution process...")
            new_population = optimizer.init_population(size = size)
            #pdb.set_trace()
            #recursion
            evolution(init_population=new_population,
                      optimizer = optimizer,
                      input_x = input_x,
                      num_generations=num_generations,
                      fittest_threshold=fittest_threshold,
                      size = size,
                      mutation_rate = mutation_rate)
        
        #generate new population
        #for now, limit parents to 5
        pop_selection = optimizer.selection(population = new_population, 
                                            fitness = fitness,
                                            num_parents=5)
        #add 20% mutation
        new_population = optimizer.mutation(population = pop_selection,
                                            rate = mutation_rate)
    return fittest_per_generation

def main():
    #load sample data 1
    df = pd.read_csv('data/sample.csv')
    df = df.iloc[:,1:]
    
    # For the optimizer
    num_generations = 30

    #set fittest threshold
    fittest_threshold = 0.005

    popsize=500

    optimizer = dtvf.GA_OPtimizeDTVF(Ts = 0.001)
    population = optimizer.init_population(size = popsize)
    
    #run evolution
    print("starting evolution...")
    fittest_per_generation=evolution(init_population=population,
                                      optimizer = optimizer,
                                      input_x = df,
                                      num_generations=num_generations,
                                      fittest_threshold=fittest_threshold,
                                      size = popsize,
                                      mutation_rate = 0.3)
    return fittest_per_generation

fittest_per_generation=main()