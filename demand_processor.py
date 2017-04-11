import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#################
#   EDIT THIS   #
excluded_day_indices = [86,248]
n_clust = 3
#################

# DTW function created by bistaumanga
# https://gist.github.com/bistaumanga/6023705
def DTW(A, B, w = 2, d = lambda x,y: (x-y)**2):
    # create the cost matrix
    A, B = np.array(A), np.array(B)
    M, N = len(A), len(B)
    cost = 999999999999 * np.ones((M, N))

    # initialize the first row and column
    cost[0, 0] = d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(A[i], B[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(A[0], B[j])
    # fill in the rest of the matrix
    for i in range(1, M):
        for j in range(max(1, i - w), min(N, i + w)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

    # find the optimal path
    n, m = N - 1, M - 1
    path = []

    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key = lambda x: cost[x[0], x[1]])
    
    path.append((0,0))
    return (cost[-1, -1])**0.5 #, path

df = pd.read_excel("C:/Users/inodu_notebook_01/Desktop/Consumo Cristalerias 2014.xlsx")
days_in_month=[31,28,31,30,31,30,31,31,30,31,30,31]
dates=[datetime(2014,m+1,d,h) for m, days in enumerate(days_in_month) \
       for d in range(1,days+1) for h in range(24)]
for i in range(8760):
    df.set_value(i,'datetime',dates[i])

excluded_days = [date(2014,1,1) + timedelta(days=d+ind) \
                 for ind,d in enumerate(excluded_day_indices)]
n_days = 365 - len(excluded_days)

df.set_index('datetime', drop=True, inplace=True)
DFlist = [group[1] for group in df.groupby(df.index.date) \
          if group[0] not in excluded_days]
data = np.empty((n_days,24))
for ind, DF in enumerate(DFlist):
    data[ind] = np.array(DF.transpose())[0]

# the linkage function may receive custom distances in a compact vector format
distance_vector = np.zeros((1,n_days*(n_days-1)/2))
a = 0
for d1 in range(n_days-1):
    for d2 in range(d1+1,n_days):
        distance_vector[0][a] = DTW(data[d1], data[d2])
        a += 1

Z=linkage(distance_vector[0], method='ward')
plt.figure()
dendrogram(Z, truncate_mode='lastp', p=n_clust, get_leaves=True)

clusters = fcluster(Z, n_clust, criterion='maxclust')
clusters_len = np.zeros((n_clust,1))
rep_curves = np.zeros((n_clust,24))
plt.figure()
for c in range(n_clust):
    for d in range(n_days):
        if clusters[d] == c+1:
            clusters_len[c] += 1
            for j in range(24):
                rep_curves[c][j] += data[d][j]
    rep_curves[c] /= clusters_len[c]
    plt.plot(rep_curves[c],linewidth=0.03*clusters_len[c])

for c in range(n_clust):
    plt.figure()
    plt.ylim((0,8500))
    for d_ind, c_ind in enumerate(clusters):
        if c_ind == c+1:
            plt.plot(data[d_ind],color='blue')
    plt.plot(rep_curves[c],color='red')