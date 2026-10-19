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

class Stats:
    def Mo(X):
        counts = {}
        for i in X:
            counts[i] = counts.get(i,0) + 1
        max_count = max(counts.values())
        modes = [i for i, count in counts.items() if count == max_count]
        return modes[0]

    def std(X):
        sum = 0
        for x_i in X:
            sum += x_i
        
        mu = sum/len(X)
        
        std_dev = 0
        for x_j in X:
            std_dev += (x_j-mu)**2
        
        return math.sqrt(std_dev/n)

    def TrueSTD(X):
        sum = 0
        for x_i in X:
            sum += x_i
        
        mu = sum/len(X)
        
        std_dev = 0
        for x_j in X:
            std_dev += (x_j-mu)**2
        
        return math.sqrt((std_dev)/(len(X) -1))

    def As(X):
        d = X - X.mean()
        return np.mean(d**3)/  np.mean(d**2)**1.5

    def Excess(X):
        d = X - X.mean()
        return (np.mean(d**4)/np.mean(d**2)**2) - 3

#1
X = np.random.normal(loc=a,scale=sigma,size=n)

#1.1
counts, _ = np.histogram(X, bins='fd')
plt.hist(X,bins='fd')
plt.show()
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

counts, bin_edges = np.histogram(X, bins='fd')
i_max = np.argmax(counts)
mode_left, mode_right = bin_edges[i_max], bin_edges[i_max + 1]
mode_count = counts[i_max]

print(f'модальный интервал: [{mode_left:.3f}; {mode_right:.3f})')
print(f'частота в нём: {mode_count}')
print(f'оценка вероятности (относительная частота): {mode_count / n:.4f}')

# оценка ФР в правой границе модального интервала
F_hat = counts[:i_max + 1].sum() / n
print(f'правая граница модального интервала: {mode_right:.4f}')
print(f'оценка F̂(m_right) = {counts[:i_max+1].sum()}/{n} = {F_hat:.4f}')

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
plt.hist(X, bins='fd', density=1, cumulative=1, alpha=0.5, label='Эмпирическая ФР')

#теоретическая функция распределения
x_axis = np.linspace(X.min(), X.max(), 300)
cdf = stats.norm.cdf(x_axis, loc=a, scale=sigma)
plt.plot(x_axis, cdf, color='red', label='Теоретическая ФР')

plt.grid(True)
plt.legend()
plt.show()

#2.5
plt.boxplot(X,vert=False,patch_artist=1,boxprops=dict(facecolor='lightblue',color='blue'),
            medianprops=dict(color='red',linewidth=2),
            flierprops=dict(marker='o',markerfacecolor='#1356DA'))
plt.show()

#3.1
print("===============3 пункт===============")
sum = 0
for i in X:
    sum += i

Xx = sum / n
X_sorted = sorted(X)
median = X_sorted[len(X_sorted) // 2]
moda = Stats.Mo(X_sorted)
gen = ((i - Xx) ** 2 for i in X)
Dx = np.sum(np.fromiter(gen,dtype=float)) / n 
gen = ((i - Xx) ** 2 for i in X)
Dx_corrected = np.sum(np.fromiter(gen,dtype=float)) / (n - 1)
std_dev = Stats.std(X)

print(f'ВС = {Xx:.3}\nВС ВС через np.mean() = {np.mean(X):.3}')
print(f'Медиана = {median:.3}\nМедиана через np.median() = {np.median(X_sorted):.3}')
print(f'Мода = {moda:.3}\nМода через scipy.stats.mode() = {stats.mode(X)[0]:.3}')
print(f'Дисперсия = {Dx:.3}\nДисперсия через np.var() = {np.var(X):.3}')
print(f'Исправленная дисперсия = {Dx_corrected:.3}\nИсправленная дисперсия через np.var() = {np.var(X, ddof=1):.3}')
print(f'Стандартное отклонение = {std_dev:.3}\nСтандарнтое отклоенние через numpy.std: {np.std(X):.3}')
print(f'Испр. стандартное отклонение = {Stats.TrueSTD(X):.3}\nИспр. стандарнтое отклоенние через numpy.std: {np.std(X, ddof=1):.3}')
print(f'Коэффициент ассиметрии = {Stats.As(X):.3}\nКоэффициент ассиметрии по numpy.assymetry = {stats.skew(X):.3}')
print(f'Эксцесс = {Stats.Excess(X):.3}\nЭксцесс по stats.kurtosis:{stats.kurtosis(X):.3}')

#3.2
N = n*60
X_big = np.random.normal(loc=a, scale=sigma, size=N)

print("===============3.2 пункт (объём выборки x60)===============")
print(f'Объём выборки N = {N}')
print(f'ВС = {np.mean(X_big):.3}')
print(f'Медиана = {np.median(X_big):.3}')
print(f'Мода = {stats.mode(X_big)[0]:.3}')
print(f'Дисперсия = {np.var(X_big):.3}')
print(f'Исправленная дисперсия = {np.var(X_big, ddof=1):.3}')
print(f'Стандартное отклонение = {np.std(X_big):.3}')
print(f'Испр. стандартное отклонение = {np.std(X_big, ddof=1):.3}')
print(f'Коэффициент ассиметрии = {stats.skew(X_big):.3}')
print(f'Эксцесс = {stats.kurtosis(X_big):.3}')

print(f'\nчерез Stats:')
print(f'ВС = {np.mean(X_big):.3}')
print(f'Испр. std = {Stats.TrueSTD(X_big):.3}')
print(f'Ассиметрия = {Stats.As(X_big):.3}')
print(f'Эксцесс = {Stats.Excess(X_big):.3}')

#3.3
print("===============3.3 пункт (анализ)===============")
print(f'Теоретические параметры: a = {a}, sigma = {sigma}, D = {sigma**2}, As = 0, Ex = 0\n')

print(f'--- выборка n = {n} ---')
print(f'ВС      = {np.mean(X):.3}   (теор. {a})')
print(f'Дисперсия = {np.var(X):.3}   (теор. {sigma**2})')
print(f'Ассиметрия = {stats.skew(X):.3}   (теор. 0)')
print(f'Эксцесс  = {stats.kurtosis(X):.3}   (теор. 0)')

print(f'\n--- выборка N = {N} ---')
print(f'ВС      = {np.mean(X_big):.3}   (теор. {a})')
print(f'Дисперсия = {np.var(X_big):.3}   (теор. {sigma**2})')
print(f'Ассиметрия = {stats.skew(X_big):.3}   (теор. 0)')
print(f'Эксцесс  = {stats.kurtosis(X_big):.3}   (теор. 0)')

err_n   = abs(np.mean(X) - a)
err_big = abs(np.mean(X_big) - a)
print(f'\n|X̄ - a| при n={n}: {err_n:.3}')
print(f'|X̄ - a| при N={N}: {err_big:.3}')