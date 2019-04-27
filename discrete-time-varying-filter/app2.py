#this code is an example use case of the Discrete-Time Variable Filter

from dtvf import DiscTimeVarFilt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def wvf_plot(wvf):
    #plot a checkweigher waveform using matplotlib
    x = np.asarray(wvf)
    n = np.arange(0, x.shape[0])
    plt.plot(x)
    plt.xlabel('samples (n)')
    plt.ylabel('weight (grams)')
    plt.title('waveform filtering using time-varying filters')
    # plt.legend()
    plt.show()    
    
def main():
    #define constants
    #construct X array (trapezoidal)
    wl = 120 #product length
    cl = 150  #conveyor belt length
    speed = 90 #belt speed
    N = 200 #simulation samples
    Ts = 0.001 #sampling period (s)
    weight = 50 #weight
    
    n = np.arange(0,N)
    
    #Instantiate DiscTimeVarFilt class
    fs = 1000
    flt=DiscTimeVarFilt(Ts = 1/fs)
    
    x=np.piecewise(n,
        [n < 0,
        (n >= 0) * (n < wl/speed*60),
        (n >= wl/speed*60) * (n < cl/speed*60),
        (n >= cl/speed*60) * (n<(cl+wl)/speed*60)],
        [0,
        lambda n: n*(weight/(wl/speed*60)),
        weight,
        lambda n: -n*weight/(wl/speed*60)+weight*(1+cl/wl)
        ])
    
    #add noise
    noise = np.random.normal(0,2,x.size)
    x_noise = x+noise
    
    y = flt.apply_filter(x_noise,xs=0, ys=0)
    
    plt.plot(n, x, label="input")
    plt.plot(n, x_noise, label="input with noise")
    plt.plot(n, y, label="filtered output")
    plt.xlabel('time (s)')
    plt.ylabel('n')
    plt.title('waveform filtering using time-varying filters')
    plt.legend()
    plt.show()
    
    #at this point, flt is using the default hyperparameters
    #initialized inside the dtvf class
    #alpha must be a number between 0 and 1
    flt=DiscTimeVarFilt(Ts = 1/fs, f_o = 200, f_inf = 1, k = 2, N_alpha = 100, alpha = 59/199)
    
    
    #load sample data 1
    df = pd.read_csv('data/belt90mmin_conv150mm/h1/ダイナミックBT2_最適PID.csv')
    df = df.iloc[:,1:]
    df.plot()
    y_1 = flt.apply_filter(df,xs=0, ys=0)
    wvf_plot(y_1)
    plt.plot(flt.wc_fxn(df.iloc[:,1]))
    
main()