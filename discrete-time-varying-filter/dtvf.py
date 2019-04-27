#Discrete-Time Varying Filter implementation
#This code is based on the filtering algorithm on the ff:
#Original paper: Dynamic mass measurement using a discrete time-variant filter
#Author: Przemyslaw Pietrzak
#https://ieeexplore.ieee.org/document/5661899

import math
import numpy as np

DEBUG_MODE = False;

if DEBUG_MODE:
    print("Welcome to DTVF : Debug Mode")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class MyException(Exception):
    pass

class DiscTimeVarFilt:
    def apply_filter(self, x, ys = 0, xs = 0):
        x=np.asarray(x)
        if(x.ndim == 1):
            x=x.reshape((x.shape[0],1))
        y=np.zeros(x.shape)
        #convert x to (length, 1) if it is unidimensional
        N=x.shape[0]
        wc = self.w_inf + (self.w_o-self.w_inf)*self.alpha**(np.arange(0,N)/self.N_alpha)
        lmb = (-1)*((wc-2*self.xi/self.Ts)/(wc+2*self.xi/self.Ts))
        lmb_s = (1-lmb)/2

        #compute filtered value
        lmb_sum=lmb+lmb_s
        y[0, :] = (lmb[0]*ys+lmb_s[0]*(x[0,:] + xs))/lmb_sum[0]
        for n in np.arange(1,N):
            #compute wc and lambda
            y[n, :] = (lmb[n]*y[n-1, :]+lmb_s[n]*(x[0] + x[n-1]))/lmb_sum[n]
        return y

    def wc_fxn(self, x):
        x=np.asarray(x)
        N=x.size
        y=np.zeros((N,1))
        wc = self.w_inf + (self.w_o-self.w_inf)*self.alpha**(np.arange(0,N)/self.N_alpha)
        return wc
    
    def print_param(self):
        attrs = vars(self)
        print('\n'.join("+%s: %s" % item for item in attrs.items()))

    def __init__(self, Ts, f_o = 200, f_inf = 0.01, k = 2, N_alpha = 150, alpha = 0.5):
        if not (alpha>0 and alpha<=1):
            raise MyException("alpha must be in the range (0,1]!")
            return None
        self.alpha = alpha
        self.N_alpha = N_alpha
        self.k = k
        self.w_o = f_o * 2 * math.pi
        self.w_inf = f_inf * 2 * math.pi
        self.Ts = Ts
        self.xi = math.sqrt(2**(k) - 1)   
    #def filter_by_gaoptim(self):

class filter_by_gaoptim:
    def __init__(self, Ts):
        # initialize an instance of the filter
        self.Ti = Ts
        self.filt = DiscTimeVarFilt(Ts)
    
    def cal_pop_fitness(self, population, meas_time, input_x, realweight, sol_per_pop):
        # This function calculates the fitness of the population
        # against the given input dataframe input_x of MxN (M samples per N waveforms)
        # meas_time is the index where 3s and mean of the waveforms is calculated
        # This function returns the solution
        
        fitness = np.zeros(len(population))
        for idx, val in enumerate(population):
            # create filter for each member
            ymem = DiscTimeVarFilt(Ts = self.Ts, **val)
            #compute 3s and variation from mean (Xbar-x)
            std3 = ymem[meas_time, :].std()*3
            xbar_x = abs(ymem[meas_time, :].mean()-realweight)
            # compute fitness: for now define fitness function as product of the two
            # the higher the value, the better
            fitness[idx] = 1/(std3*xbar_x)
        # normalize fitness score
        return fitness/np.sum(fitness)        
    
    def selection(population, fitness, num_parents):
        # Pick members from the current generation
        # to be the parents in the next generation
        # crossover & mutation (in percentage)
        
    
    def make_member(self, constr = {
            "f_o": (31,200),
            "f_inf": (30, 0.01),
            "k": (2, 3),
            "N_alpha" : (50,400),
            "alpha": (0, 1)}):
        # this function creates a member of the population
        # with each member being a dictionary of hyperparameters
        # This returns a dictionary with random values for hyperparameters
        
        # The constraints on hyperparameters are in the "constraints" dictionary
        # that indicates the max and minimum allowable values for each hyperparameter
        
        hyperparams = {
                "f_o": np.random.uniform(*constr["f_o"]),
                "f_inf": np.random.uniform(*constr["f_inf"]),
                "k": np.random.uniform(*constr["k"]),
                "N_alpha" : np.random.uniform(*constr["N_alpha"]),
                "alpha": np.random.uniform(*constr["alpha"])
                }
        return hyperparams
    
        
    def init_population(self, size):
        population = []
        for i in range(0, size):
            population.append(make_member())    
        return population    