#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:52:32 2019

@author: jay
"""
import dtvf
import numpy as np
import pandas as pd

#def main():
optimizer=dtvf.GA_OPtimizeDTVF(Ts = 0.001)
population = optimizer.init_population(size = 100)

#load sample data 1
df = pd.read_csv('data/sample.csv')
df = df.iloc[:,1:]

fitness = optimizer.cal_pop_fitness(population = population, 
                                    meas_time = 99, 
                                    input_x = df, 
                                    realweight = 22.51, 
                                    sol_per_pop = 10)

#select 5 fittest members
fittest_idx = np.argsort(fitness)[-5:]

fittest_member=[]
for idx, mem in enumerate(fittest_idx):
    fittest_member.append(population[mem])

#try for the fittest member
flt=dtvf.DiscTimeVarFilt(Ts =0.001,**fittest_member[4])
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