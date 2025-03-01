import networkx as nx
from networkx.classes.function import path_weight
import math
import numpy as np
import matplotlib.pyplot as plt
import random

# 2d grid graph
N = 20   # 10, 20, 40
G = nx.grid_2d_graph(N, N)
mean_edge_wt = 0.5

for edge in G.edges():
    G.edges[edge]['weight'] = random.uniform(0, 1)

node_list = list(G.nodes())
cardinality = len(node_list)*(len(node_list)-1)/2

V = list()
Node_pair_bins = [[], [], [], []]

for i in range(len(node_list)):
    for j in range(i+1, len(node_list)):
        src = node_list[i]
        tgt = node_list[j]
        dist = nx.shortest_path_length(G, src, tgt, 'weight')
        if dist > 0:
            V.append(dist)

quantiles = [np.percentile(V, 25), np.percentile(V, 50), np.percentile(V, 75)]

for i in range(len(node_list)):
    for j in range(i+1, len(node_list)):
        src = node_list[i]
        tgt = node_list[j]
        dist = nx.shortest_path_length(G, src, tgt, 'weight')
        t = sum((dist > quantiles[0], dist > quantiles[1], dist > quantiles[2]))
        if dist > 0:
            Node_pair_bins[t].append([src, tgt])

counts = [len(Node_pair_bins[0]), len(Node_pair_bins[1]), len(Node_pair_bins[2]), len(Node_pair_bins[3])]
density = [counts[i]/sum(counts) for i in range(len(counts))]

# random sampling according to estimated densities from each category
random_sample = list()
sample_size = 100
category_sz = [math.ceil(sample_size*density[i]) for i in range(len(counts))]

for k in range(len(density)):
    subsample = list()
    while len(subsample) < category_sz[k]:
        draw = np.random.randint(len(Node_pair_bins[k]))        # sampling step
        pair = Node_pair_bins[k][draw]
        if pair not in subsample:
            subsample.append(pair)
            Node_pair_bins[k].remove(pair)
    random_sample = random_sample + subsample


# privacy parameters
std = 1*mean_edge_wt   # 0.2, 0.5, 1
mu = 0

graph_instances = 25

Ratio = [0, 0, 0, 0, 0, 0, 0]
Ratio_cat = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

for count in range(graph_instances):

    # generate a random graph instance with random edge weights
    for edge in G.edges():
        G.edges[edge]['weight'] = random.uniform(0, 1)

    local_V = list()

    # obtain SP lengths for all node-pairs in the sample
    for node_pair in random_sample:
        source = list(node_pair)[0]
        target = list(node_pair)[1]
        sp = nx.shortest_path_length(G, source, target, 'weight')
        local_V.append(sp)

    # generate private copies of graph and run computations
    for itr in range(100):
        private_G = G.copy()

        for edge in private_G.edges():
            noise = np.random.normal(mu, std, 1)
            t = max(0, private_G.edges[edge]['weight'] + float(noise))
            private_G.edges[edge]['weight'] = t

        private_V = list()

        for node_pair in random_sample:
            source = list(node_pair)[0]
            target = list(node_pair)[1]
            path = nx.shortest_path(private_G, source, target, 'weight')
            sp = path_weight(G, path, 'weight')
            private_V.append(sp)

        for j in range(len(local_V)):
            val1 = local_V[j]
            val2 = private_V[j]
            frac = val2/val1
            index_list = [0, 1, 2, 3, 4, 5, 6]
            index = index_list[sum((frac > 1, frac > 1.1, frac > 1.2, frac > 1.4, frac > 1.6, frac > 2))]
            row_list = [0, 1, 2, 3]
            row_index = row_list[sum((val1 > quantiles[0], val1 > quantiles[1], val1 > quantiles[2]))]
            Ratio_cat[row_index][index] += 1
            Ratio[index] += 1


Tot = sum(Ratio)
Prob = [Ratio[i]/Tot for i in range(len(Ratio))]
Prob_cat = []
for i in range(4):
    row = []
    for j in range(7):
        item = Ratio_cat[i][j]/sum(Ratio_cat[i])
        row.append(item)
    Prob_cat.append(row)


plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots()
categories = ['0', '0-10', '10-20', '20-40', '40-60', '60-100', '>100']
width = 0.15

x_1 = [x-1.5*width for x in range(len(Prob_cat[0]))]
x_2 = [x-0.5*width for x in range(len(Prob_cat[1]))]
x_3 = [x+0.5*width for x in range(len(Prob_cat[2]))]
x_4 = [x+1.5*width for x in range(len(Prob_cat[3]))]

ax.bar(x_1, Prob_cat[0], width, color='blue', label=r'Category 1')
ax.bar(x_2, Prob_cat[1], width, color='olive', label=r'Category 2')
ax.bar(x_3, Prob_cat[2], width, color='orange', label=r'Category 3')
ax.bar(x_4, Prob_cat[3], width, color='red', label=r'Category 4')
ax.xaxis.set_tick_params(labelsize=14.5)
ax.yaxis.set_tick_params(labelsize=17)

ax.grid(True)
ax.plot(categories, [0, 0, 0, 0, 0, 0, 0])
ax.set_ylim(0, 1)
# ax.set_title('2D grid (10x10): Shortest path change statistics')
ax.set_xlabel('Relative Bias (%)', fontsize=19)
ax.set_ylabel('Empirical Probability', fontsize=19)

plt.show()

