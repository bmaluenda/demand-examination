import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

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

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return (LB_sum**0.5)

import random

def k_means_clust(data,num_clust,num_iter,w):
    centroids=random.sample(data,num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print counter
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTW(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]
        print assignments
        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]
    return centroids

df = pd.read_excel("C:/Users/inodu_notebook_01/Desktop/Consumo Cristalerias 2014.xlsx")
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
data = np.empty((365,24))
for ind, DF in enumerate(DFlist):
    data[ind] = np.array(DF.transpose())[0]
centroids=k_means_clust(data,num_clust=2,num_iter=10,w=2)