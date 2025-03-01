import math
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm

matplotlib.rcParams['axes.linewidth'] = 2

delta = 0.01
K = 1/math.sqrt(2*math.log(1.25/delta))
delf = 1
eps = 1
sigma = delf/(K*eps)

gamma = 0.05
cardinality = [2, 5, 10, 20]
N = [5, 10, 15, 20, 25]

for c in cardinality:
    delta_bound = []
    for n in N:
        z_score = norm.ppf(1-gamma/c)
        val = math.sqrt(2*n)*z_score*sigma
        delta_bound.append(val)

    if c == 2:
        plt.plot(N, delta_bound, linestyle='dashdot', linewidth=2, label=r'$|\mathcal{P}_{ij}| = 2$')
    elif c == 5:
        plt.plot(N, delta_bound, linestyle='dotted', linewidth=2, label='$|\mathcal{P}_{ij}| = 5$')
    elif c == 10:
        plt.plot(N, delta_bound, linestyle='dashed', linewidth=2, label='$|\mathcal{P}_{ij}| = 10$')
    else:
        plt.plot(N, delta_bound, linestyle='solid', linewidth=2, label='$|\mathcal{P}_{ij}| = 20$')


plt.xticks([5, 10, 15, 20, 25], size=9, weight='bold')
plt.yticks(size=9, weight='bold')
plt.xlim([5, 25])
plt.ylim([10, 70])
plt.grid(True)
plt.ylabel(r'Upper bound on Bias', fontsize=15)
plt.xlabel(r'Max number of edges S', fontsize=15)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.legend(loc='lower right', frameon=False, prop={'size': 15.0})

plt.show()
