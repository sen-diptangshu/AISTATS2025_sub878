import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm

matplotlib.rcParams['axes.linewidth'] = 2

# dependence on gap \alpha, eps and S
delta = 0.01
K = 1/math.sqrt(2*math.log(1.25/delta))
eps = 1

gap = 15
for S in np.arange(5, 25, 5):
    val = []
    e = np.arange(0.1, 2.1, 0.1)
    for delf in e:
        val.append(1-norm.cdf(K*gap*eps/(delf*math.sqrt(S))))
    if S == 5:
        plt.plot(e, val, linestyle='dashdot', linewidth=2, label='|S| = 5')
    elif S == 10:
        plt.plot(e, val, linestyle='dotted', linewidth=2, label='|S| = 10')
    elif S == 15:
        plt.plot(e, val, linestyle='dashed', linewidth=2, label='|S| = 15')
    else:
        plt.plot(e, val, linestyle='solid', linewidth=2, label='|S| = 20')

plt.xticks(size=9, weight='bold')
plt.yticks(size=9, weight='bold')
plt.xlim([0.1, 2])
plt.ylim([-0.02, 0.4])
plt.grid(True)
plt.ylabel(r'Probability $q$', fontsize=15)
plt.xlabel(r'Sensitivity $\Delta f$', fontsize=15)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.legend(loc='upper left', frameon=False, prop={'size': 15.0})

plt.show()
