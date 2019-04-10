#this code is an example use case of the Discrete-Time Variable Filter

from dtvf import DiscTimeVarFilt
import numpy as np
import matplotlib.pyplot as plt

flt=DiscTimeVarFilt(Ts = 0.001)

#construct X array
x = np.random.rand(150,1)
y = flt.apply_filter(x)

t = np.arange(0,x.size)

plt.plot(t,x,t,y)
plt.show()
