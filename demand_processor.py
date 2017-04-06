import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# DTW function created by bistaumanga
# https://gist.github.com/bistaumanga/6023705
def DTW(A, B, window = 2, d = lambda x,y: abs(x-y)):
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
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

    # find the optimal path
    n, m = N - 1, M - 1
    path = []

    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key = lambda x: cost[x[0], x[1]])
    
    path.append((0,0))
    return cost[-1, -1], path

df = pd.read_excel("C:/Users/inodu_desk_01/Desktop/Sergio/Proyecto gestion demanda/Consumo Cristalerias 2014.xlsx")
days=[31,28,31,30,31,30,31,31,30,31,30,31]
dates=[]
for m in [1,2,3,4,5,6,7,8,9,10,11,12]:
     for d in range(1,days[m-1]+1):
         for h in range(24):
             dates.append(datetime(2014,m,d,h))
for i in range(8760):
    df.set_value(i,'datetime',dates[i])

df.set_index('datetime',drop=True,inplace=True)
DFlist = [group[1] for group in df.groupby(df.index.date)]