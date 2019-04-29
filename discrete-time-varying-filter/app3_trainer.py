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

import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dtvf

#import pdb #breakpoint generator

def wvf_plot(wvf):
    """plot a checkweigher waveform using matplotlib"""
    x = np.asarray(wvf)
    plt.plot(x)
    plt.xlabel('samples (n)')
    plt.ylabel('weight (grams)')
    plt.title('waveform filtering using time-varying filters')
    plt.legend()
    plt.show()    
    

def evolution(init_population, input_x, optimizer, num_generations, fittest_threshold, size, mutation_rate=0.2):
    
    #define re-evolve variable
    re_evolve=True
    new_population = init_population
    fittest_per_generation=[]
    fittest_scores=np.zeros((num_generations,3))
    while (re_evolve):
        re_evolve=False #this variable changes to True if the evolution fails
        for i in range(num_generations):
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
            fittest_per_generation.append((new_population[fittest_idx[0]]))
            fittest_scores[i,:] = fitness[fittest_idx[0],:]
            
            # Check fittest member
            # If fittest member has a fitness value of less than a certain defined threshold,
            # restart the whole evolution process
            
            if (fitness[fittest_idx[0],0] < fittest_threshold):
                #restart evolution process
                print("Encountered max(fitness)=",fitness[fittest_idx[0],0],"<",fittest_threshold,". Restarting evolution process...")
                #pdb.set_trace()
                re_evolve = True
                break
            
            #generate new population
            #for now, limit parents to 5
            pop_selection = optimizer.selection(population = new_population, 
                                                fitness = fitness,
                                                num_parents=5)
            #add 20% mutation
            new_population = optimizer.mutation(population = pop_selection,
                                                rate = mutation_rate)
        if(re_evolve==1):
            fittest_per_generation=[]
            fittest_scores=np.zeros((num_generations,3))
            new_population = optimizer.init_population(size = size)            
    return fittest_scores, fittest_per_generation
        

def main():
    #load sample data 1
    df = pd.read_csv('data/sample.csv')
    df = df.iloc[:,1:]
    
    # For the optimizer
    num_generations = 20

    #set fittest threshold
    fittest_threshold = 0.005

    popsize=700

    optimizer = dtvf.DTVFOptimizerByGA(ts = 0.001)
    population = optimizer.init_population(size = popsize)
    
    #run evolution
    print("starting evolution...")
    return evolution(init_population = population,
                                      optimizer = optimizer,
                                      input_x = df,
                                      num_generations = num_generations,
                                      fittest_threshold = fittest_threshold,
                                      size = popsize,
                                      mutation_rate = 0.3)
    
fittest_scores, fittest_per_generation=main()

def plot_learning_curve(fittest_scores):
    #plot data
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('generation')
    ax1.set_ylabel('3s', color=color)
    ax1.plot(np.arange(fittest_scores.shape[0]), fittest_scores[:,1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0,0.5])
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('|Xbar-x|', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.arange(fittest_scores.shape[0]), fittest_scores[:,2], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0,1])
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")

#write fittest_scores and fittest_per_generation
def write_to_csv(fittest_scores, fittest_per_generation, filename="out_"+timestr+".csv"):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['f_o', 'f_inf', 'k', 'alpha', 'n_alpha', 'fit-score', '3s', 'Xbar-x']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx,val in enumerate(fittest_per_generation):
            val['fit-score']=fittest_scores[idx,0]
            val['3s']=fittest_scores[idx,1]
            val['Xbar-x']=fittest_scores[idx,2]
            writer.writerow(val)