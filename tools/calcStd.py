# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import utils.defaults as defaults
x = np.array([1, 2, 3, 4, 5])
y = np.power(x, 2) # Effectively y = x**2
e = np.array([1.5, 2.6, 3.7, 4.6, 10])
samples  = np.load(defaults.file_dataset51)
labels = np.load(defaults.file_labels51)
samples_12  = np.load(defaults.file_dataset12NoLog)


mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

ax.set_title("51 distancies")
ax.set_xlabel("n-distancia")
ax.set_ylabel("Distancia entre els dos punts")

plt.xlim([-1,52])

x  = np.arange(len(samples[0]))
colmeans = samples.mean(axis=0)
colstd = samples.std(axis=0)
print colstd[0]
plt.errorbar(x, colmeans, colstd, linestyle='None', marker='^', color='red')
markers = ['o','x','<', '>','+','.']
colors = ['b', 'g','c', 'm', 'y', 'k']
acum = 0.1
for i in xrange(defaults.CLASS_N):
    print i
    colmeans = samples[labels==i].mean(axis=0)
    colstd = samples[labels==i].std(axis=0)
    x= x+acum
    plt.errorbar(x, colmeans, colstd, linestyle='None', marker=markers[i], color=colors[i])

colmeans = samples_12.mean(axis=0)
colstd = samples_12.std(axis=0)
x  = np.arange(len(samples_12[0]))
print colstd[0]
plt.errorbar(x, colmeans, colstd, linestyle='None', marker='H', color='gold')

plt.show()

