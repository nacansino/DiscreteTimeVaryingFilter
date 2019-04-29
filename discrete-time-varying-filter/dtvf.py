#Discrete-Time Varying Filter implementation
#This code is based on the filtering algorithm on the ff:
#Original paper: Dynamic mass measurement using a discrete time-variant filter
#Author: Przemyslaw Pietrzak
#https://ieeexplore.ieee.org/document/5661899

import numpy as np
import itertools as it
import random

DEBUG_MODE = False;

if DEBUG_MODE:
    print("Welcome to DTVF : Debug Mode")

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def elu(x, alpha):
    if (x >= 0):
        return x
    else:
        return alpha * (np.e**x -1)
    
def gaussff(x,y,sx,sy):
    return 100*np.exp(-1*((x**2/(2*sx**2))+(y**2/(2*sy**2))))
    
def exp(x, tau):
    #exponential fitness fxn
    return np.e**(x/tau)

def neg_relu(x, inf_point):
    if (x <= inf_point):
        return inf_point-x
    else:
        return 0

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
        #no need for normalization)
        y[0, :] = lmb[0]*ys+lmb_s[0]*(x[0,:] + xs)
        for n in np.arange(1,N):
            #compute wc and lambda
            y[n, :] = lmb[n]*y[n-1, :]+lmb_s[n]*(x[n, :] + x[n-1, :])
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
        self.w_o = f_o * 2 * np.pi
        self.w_inf = f_inf * 2 * np.pi
        self.Ts = Ts
        self.xi = np.sqrt(2**(k) - 1)   
    #def filter_by_gaoptim(self):

class GA_OPtimizeDTVF:
    def __init__(self, Ts):
        # the only information that the optimizer needed to know
        # is the sampling time Ts (1/fs)
        self.Ts = Ts
        # self.optim_filt = DiscTimeVarFilt(Ts)
    
    def genCDF(x):
        # accept x as 1-d input.
        # generates 1-d CDF
        cdf = np.zeros(x.shape[0])
        cdf[0]=x[0]
        for i in range(1,x.shape[0]):
            cdf[i]=cdf[i-1]+x[i]
        return cdf
    
    def cal_pop_fitness(self, population, meas_time, input_x, realweight):
        # This function calculates the fitness of the population
        # against the given input dataframe input_x of MxN (M samples per N waveforms)
        # meas_time is the index where 3s and mean of the waveforms is calculated
        # This function returns the solution
        
        fitness = np.zeros((len(population),3))
        for idx, val in enumerate(population):
            # create filter for each member
            param_mem = DiscTimeVarFilt(Ts = self.Ts, **val)
            #apply filter created by the current member to the input
            ymem = param_mem.apply_filter(input_x, ys=0, xs=0)
            #compute 3s and variation from mean (Xbar-x)
            std3 = ymem[meas_time, :].std()*3
            xbar_x = abs(ymem[meas_time, :].mean()-realweight)
            # Compute fitness 
            # The fitness function is a lambda fxn.
            # It is either supplied as an argument or it defaults to gaussff()
            fitness[idx, 1] = std3
            fitness[idx, 2] = xbar_x
        fitness[:,0]=gaussff(fitness[:,1],fitness[:,2],0.4,0.3)
        # normalize fitness score
        return fitness
    
    def selection(self, population, fitness, num_parents):
        # Pick members from the current generation
        # to be the parents in the next generation
        # crossover & mutation (in percentage)
        # Selection formula from CDF:
        # np.min(np.argwhere(cdf>np.random.rand()))
        
        #pick potential parents from the seed
        #for now get the top performing members
        N = len(population)
        fittest_idx = np.argsort(fitness[:,1],)[:num_parents]
        parents=[population[i] for i in fittest_idx]
        
        #produce offspring
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
        newpop_params = random.sample(f_alpha_Nalpha_k, k=int(0.80*N))
        #convert newpop(expressed in tuples) to list of dictionaries
        newpop = []
        for i,val in enumerate(newpop_params):
            newpop.append({"f_o": val[0],
                         "f_inf": val[1],
                         "k": val[4],
                         "N_alpha" : val[3],
                         "alpha": val[2]})
#        for i in range(0, num_parents):
#            cdf = self.genCDF(fitness_score)
#            draw_idx = np.min(np.argwhere(cdf>np.random.rand()))
#            #append it to parents
#            parents.append(population[draw_idx])
#            #
        # fill the remaining 10% with new randomly spawned members
        newpop_newmems = self.init_population(N-len(newpop))
        newpop += newpop_newmems
        return newpop
    
    def mutation(self, population, rate = 0.2):
        #varies the population's parameters by the amount of indicated rate
        mutated_population = []
        for i, mem in enumerate(population):
            # Python does not allow updating dictionary while iterating over it
            # create new dictionary (member) instead
            newmem = dict()
            for key, val in mem.items():
                # increment/decrement the parameter by +=rate(%) amount
                newval = val + val*rate*(np.random.rand()-0.5)*2
                if (key != "k"):
                    if (key == "alpha"):
                        newval = min(max(0.0000001, newval),1)
                    newmem[key] = newval
                else:
                    newmem[key] = val
            mutated_population.append(newmem)
        return mutated_population
    # usage: 
    def make_member(self = None, constr = {
            "f_o": (31,200),
            "f_inf": (30, 0.01),
            "k": (2, 4),
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
                "k": np.random.randint(*constr["k"]),
                "N_alpha" : np.random.uniform(*constr["N_alpha"]),
                "alpha": np.random.uniform(*constr["alpha"])
                }
        return hyperparams
    
    def init_population(self, size):
        population = []
        for i in range(0, size):
            population.append(self.make_member())    
        return population    