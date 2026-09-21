import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sc
from scipy import stats

#Вариант 6
EPS = 15e-3
a , sigma_quadro = -1,2 
n = 125
k = 5

X = np.random.normal(loc=a, scale=sigma_quadro, size=n)
plt.grid(True)
n,bins, patches = plt.hist(X, bins='fd', density=1,cumulative=0)
#plt.hist(X, bins='scott')
#plt.hist(X,bins='sturges')
hist_ret, z = np.histogram(X, bins=k, density=1)
print(f"hist_ret = {np.sum(hist_ret)*np.diff(z)[0]}, z = {z}")
print(f"n = {n[-1]},sum n = {np.sum(n)}")
plt.show()