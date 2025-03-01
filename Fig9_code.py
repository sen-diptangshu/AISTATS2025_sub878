import networkx as nx
from networkx.classes.function import path_weight
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import random

matplotlib.rcParams['axes.linewidth'] = 2

N = 21
G = nx.wheel_graph(N)

for edge in G.edges():
    G.edges[edge]['weight'] = random.uniform(0, 1)

node_list = list(G.nodes())

source1 = 1
target1 = 10

source2 = 0
target2 = 1

list1 = list()
list2 = list()
s1 = 0
s2 = 0

for path in nx.all_simple_paths(G, source1, target1):
    w = path_weight(G, path, 'weight')
    if len(path)-1 > s1:
        s1 = len(path)-1
    list1.append(w)

for path in nx.all_simple_paths(G, source2, target2):
    v = path_weight(G, path, 'weight')
    if len(path)-1 > s2:
        s2 = len(path)-1
    list2.append(v)

list1.sort()
list2.sort()

beta1 = [list1[i]-list1[0] for i in range(1, len(list1))]
beta2 = [list2[i]-list2[0] for i in range(1, len(list2))]
beta1[0] = 0
beta2[0] = 0

std = 0.3

x = []
y = []
z = []

for b in np.arange(0, 10.5, 0.5):
    x.append(b)
    card1 = sum((beta1[i] >= b) for i in range(len(beta1)))
    card2 = sum((beta2[i] >= b) for i in range(len(beta2)))
    prob1 = min(1, card1 * (1 - norm.cdf(b / (std * math.sqrt(s1)))))
    prob2 = min(1, card2 * (1 - norm.cdf(b / (std * math.sqrt(s2)))))
    y.append(prob1)
    z.append(prob2)


plt.plot(x, y, color='blue', linewidth=2, label=r'Source = 1, Target = 10')
plt.plot(x, z, color='red', linewidth=2, label=r'Source = 0, Target = 1')

plt.xticks([0, 2, 4, 6, 8, 10], size=9, weight='bold')
plt.yticks(size=9, weight='bold')
plt.xlim([0, 10])
plt.ylim([-0.05, 1.05])
plt.grid(True)
plt.xlabel(r'$\beta$', fontsize=15)
plt.ylabel(r'Upper bound on $q_{\beta}$', fontsize=15)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.legend(loc='upper right', frameon=False, prop={'size': 12.0})

plt.show()





