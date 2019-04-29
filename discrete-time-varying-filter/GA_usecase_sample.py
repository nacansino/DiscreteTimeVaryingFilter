#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:52:32 2019

@author: jay

DO NOT RUN THIS NOTEBOOK BY ITSELF. THIS IS JUST A REPOSITORY OF SAMPLE CODES DURING DEVELOPMENT
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
    
#compute 3s and Xbar-x of 5 fittest members


fittest_member=[]
for idx, mem in enumerate(fittest_idx):
    fittest_member.append(population[mem])

#try for the fittest member (lowest 3s)
flt=dtvf.DiscTimeVarFilt(Ts =0.001,**fittest_member[0])
y_1 = flt.apply_filter(df,xs=0, ys=0)

# plot using plotly
# the ff. code won't run on sypder :p
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode()

traces = []
for idx in range(0, y_1.shape[1]):
    traces.append(go.Scatter(y = y_1[:, idx],
                            mode = 'lines',
                            name = 'wvf'+str(idx+1)
                            )
                 )

iplot(traces)

#main()

#testing code for main filter
df = pd.read_csv('data/sample.csv')
df = df.iloc[:,1:]
x=np.asarray(df)
if(x.ndim == 1):
    x=x.reshape((x.shape[0],1))
y=np.zeros(x.shape)

Ts=0.001
w_o = 114.62473467351174
w_inf = 6.891812475300544
k = 2,
N_alpha = 53.42737302270834,
alpha = 0.1600688326505907
xi = 1.7

N=x.shape[0]
wc = w_inf + (w_o-w_inf)*alpha**(np.arange(0,N)/N_alpha)
lmb = (-1)*((wc-2*xi/Ts)/(wc+2*xi/Ts))
lmb_s = (1-lmb)/2

#compute filtered value
lmb_sum=lmb+lmb_s
y[0, :] = (lmb[0]*ys+lmb_s[0]*(x[0,:] + xs))/lmb_sum[0]
for n in np.arange(1,N):
    #compute wc and lambda
    y[n, :] = (lmb[n]*y[n-1, :]+lmb_s[n]*(x[n, :] + x[n-1, :]))/lmb_sum[n]
    
    
#"zip" codes
import itertools as it
import random

f = []
k = []
N_alpha = []
alpha = []
for idx, val in enumerate(parents):
    f.append((val["f_o"], val["f_inf"]))
    k.append(val["k"])
    N_alpha.append(val["N_alpha"])
    alpha.append(val["alpha"])
    
f_alpha = [(*val[0],val[1]) for idx, val in enumerate(it.product(f,alpha))]
f_alpha_Nalpha = [(*val[0],val[1]) for idx, val in enumerate(it.product(f_alpha,N_alpha))]
f_alpha_Nalpha_k = [(*val[0],val[1]) for idx, val in enumerate(it.product(f_alpha_Nalpha,k))]

#get 90% from parents
newpop_params = random.sample(f_alpha_Nalpha_k, k=80)
#convert newpop(expressed in tuples) to list of dictionaries
newpop = []
for i,val in enumerate(newpop_params):
    newpop.append({"f_o": val[0],
                 "f_inf": val[1],
                 "k": val[4],
                 "N_alpha" : val[3],
                 "alpha": val[2]})

#spawn 5 members from random element 