import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sc
from scipy import stats

#Вариант 6
EPS = 15e-3
a , sigma = -1,4
n = 125
k = 5

#1
X = np.random.normal(loc=a,scale=sigma,size=n)

#1.1
counts, _ = np.histogram(X, bins='fd') 
print(f"интервалы = {counts}")

#1.2
print(f'сумма абсолютных частот = {np.sum(counts)}')

#1.3
rel_counts = counts / np.sum(counts)
print(f'относительные частоты = {rel_counts}')

#1.4
print(f'сумма относительных частот = {np.sum(rel_counts)}')

#2.1
bins_list = list(range(2,11) ) + list(range(15,26))
n_plots = len(bins_list)
n_cols = 4
n_rows = math.ceil(n_plots / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
axes = axes.flatten()

for ax, b in zip(axes, bins_list):
    ax.hist(X, bins=b, density=True)
    ax.set_title(f'bins = {b}')
    ax.grid(True)
    ax.set_xlim(X.min() - 1, X.max() + 1)
    ax.set_ylim(0, 0.2)

plt.grid(True)
plt.tight_layout()
plt.show()

#2.2
plt.title('Гистограмма абсолютных частот')
plt.hist(X, bins='fd', density=0)
plt.grid(True)
plt.show()

#2.3
plt.title('Гистограмма относительных частот')
plt.hist(X, bins='fd', density=1)

x_axis = np.linspace(X.min() - 1, X.max() + 1, 300)
pdf = stats.norm.pdf(x_axis, loc=a, scale=sigma)
plt.plot(x_axis, pdf,color='red', label='PDF')

plt.grid(True)
plt.show()

#2.4
plt.figure(figsize=(8, 5))
plt.title('Эмпирическая и теоретическая функции распределения')

#эмпирическая функция распределения
plt.hist(X, bins='fd', density=True, cumulative=True, alpha=0.5, label='Эмпирическая ФР')

#теоретическая функция распределения
x_axis = np.linspace(X.min() - 1, X.max() + 1, 300)
cdf = stats.norm.cdf(x_axis, loc=a, scale=sigma)
plt.plot(x_axis, cdf, color='red', label='Теоретическая ФР')

plt.grid(True)
plt.legend()
plt.show()