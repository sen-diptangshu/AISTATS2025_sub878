import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
import scipy.stats as st

matplotlib.rcParams['axes.linewidth'] = 2
mu = 0
sigma = 1

x = np.linspace(mu - 5*sigma, mu + 5*sigma, 200)
plt.plot(x, st.norm.pdf(x, mu, sigma))
plt.grid(True)
plt.xlim([-5, 5])
plt.ylabel(r'f(z)', fontsize=15)
plt.xlabel('z', fontsize=15)
plt.xticks([-5, -3, -1, 1, 3, 5], size=9, weight='bold')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

cardinality = [2, 5, 20]
gamma = 0.05
for c in cardinality:
    z = st.norm.ppf(1-gamma/c)
    if c == 2:
        plt.axvline(x=z, color='red', linestyle='-', linewidth=2, label=r'$|\mathcal{P}_{ij}| = 2$')
    elif c == 5:
        plt.axvline(x=z, color='green', linestyle='--', linewidth=2, label=r'$|\mathcal{P}_{ij}| = 5$')
    else:
        plt.axvline(x=z, color='blue', linestyle='-.', linewidth=2, label=r'$|\mathcal{P}_{ij}| = 20$')


plt.legend(loc='upper left', frameon=False, prop={'size': 15.0})
plt.show()



