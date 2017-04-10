# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:46:22 2017

@author: inodu_desk_01
"""

import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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

#df = pd.read_csv("C:/Users/inodu_desk_01/Documents/Python Scripts/Power-Networks-LCL-June2015(withAcornGps)v2_1.csv")
#days_in_month=[31,28,31,30,31,30,31,31,30,31,30,31]
##dates=[]
#
#master=pd.DataFrame(columns=['LCLid','Month','Day','Year','Hour','KWH/hh (per half hour) ','ACORN'])
#masterfilt=pd.DataFrame(columns=['LCLid','Month','Day','Year','Hour','KWH/hh (per half hour) ','ACORN'])
#horas=['12:30:00 AM', '1:00:00 AM', '1:30:00 AM', '2:00:00 AM', '2:30:00 AM', '3:00:00 AM', '3:30:00 AM', '4:00:00 AM', '4:30:00 AM', '5:00:00 AM', '5:30:00 AM', '6:00:00 AM', '6:30:00 AM', '7:00:00 AM', '7:30:00 AM', '8:00:00 AM', '8:30:00 AM', '9:00:00 AM', '9:30:00 AM', '10:00:00 AM', '10:30:00 AM','11:00:00 AM', '11:30:00 AM', '12:00:00 PM', '12:30:00 PM', '1:00:00 PM', '1:30:00 PM', '2:00:00 PM', '2:30:00 PM', '3:00:00 PM', '3:30:00 PM', '4:00:00 PM', '4:30:00 PM', '5:00:00 PM', '5:30:00 PM', '6:00:00 PM', '6:30:00 PM', '7:00:00 PM', '7:30:00 PM', '8:00:00 PM', '8:30:00 PM', '9:00:00 PM', '9:30:00 PM', '10:00:00 PM', '10:30:00 PM', '11:00:00 PM', '11:30:00 PM' ]
#
# # Generate lists of unique elements (London case)
#years=df.Year.unique().tolist()  
#df1= df[df['Year']==2013] #all values for year 2013,
#hours=df1.Hour.unique().tolist()
#username=df1.LCLid.unique().tolist()  
#dates=df1.Date.unique().tolist()
#
## Generate master df by concat
#master=master.append(df1)
#master=master.reset_index(drop=True)
#
#for m in range(1,13):
#     for d in range(1,days_in_month[m-1]+1):
#         for h in range(24):
#             dates.append(datetime(2014,m,d,h))
#for i in range(8760):
#    df.set_value(i,'datetime',dates[i])
#
#df.set_index('datetime', drop=True, inplace=True)
#DFlist = [group[1] for group in df.groupby(df.index.date)]
#data = np.empty((365,24))
#for ind, DF in enumerate(DFlist):
#    data[ind] = np.array(DF.transpose())[0]

numfiles2read = 2 # between 1 and 136 (+1 from the actual number of files that we want to read)

filenames={}
#strname='Power-Networks-LCL-June2015(withAcornGps)v2_16.csv'                                     
filenum=range(1,numfiles2read)  

for z in range(len(filenum)):
    strname='Power-Networks-LCL-June2015(withAcornGps)v2_' + str(filenum[z]) +'.csv'
    filenames[z]=strname

master=pd.DataFrame(columns=['LCLid','Month','Day','Year','Hour','KWH/hh (per half hour) ','ACORN'])
masterfilt=pd.DataFrame(columns=['LCLid','Month','Day','Year','Hour','KWH/hh (per half hour) ','ACORN'])
horas=['12:30:00 AM', '1:00:00 AM', '1:30:00 AM', '2:00:00 AM', '2:30:00 AM', '3:00:00 AM', '3:30:00 AM', '4:00:00 AM', '4:30:00 AM', '5:00:00 AM', '5:30:00 AM', '6:00:00 AM', '6:30:00 AM', '7:00:00 AM', '7:30:00 AM', '8:00:00 AM', '8:30:00 AM', '9:00:00 AM', '9:30:00 AM', '10:00:00 AM', '10:30:00 AM','11:00:00 AM', '11:30:00 AM', '12:00:00 PM', '12:30:00 PM', '1:00:00 PM', '1:30:00 PM', '2:00:00 PM', '2:30:00 PM', '3:00:00 PM', '3:30:00 PM', '4:00:00 PM', '4:30:00 PM', '5:00:00 PM', '5:30:00 PM', '6:00:00 PM', '6:30:00 PM', '7:00:00 PM', '7:30:00 PM', '8:00:00 PM', '8:30:00 PM', '9:00:00 PM', '9:30:00 PM', '10:00:00 PM', '10:30:00 PM', '11:00:00 PM', '11:30:00 PM' ]
  
for currentfile in range(len(filenum)):
    df = pd.read_csv(filenames[currentfile], usecols=[0,3,4,5,6,7,8,9]) #need to figure out how to call all the names
    df=df[df['KWH/hh (per half hour) ']!='Null']  
    df=df[df['Hour']!='12:00:00 AM']
    df['KWH/hh (per half hour) ']=df['KWH/hh (per half hour) '].astype('float')
    df=df[df['KWH/hh (per half hour) '] < 10]
    
                  
    # Generate lists of unique elements
    years=df.Year.unique().tolist()  
    df1= df[df['Year']==2013] #all values for year 2012,
    hours=df1.Hour.unique().tolist()
    username=df1.LCLid.unique().tolist()  
    dates=df1.Date.unique().tolist()
    # Generate master df by concat
    master=master.append(df1)
  #  master=master[master['KWH/hh (per half hour) ']!='Null']
    master=master.reset_index(drop=True)

masterdates=master.Date.unique().tolist()   
masteruser=master.LCLid.unique().tolist() 
master['KWH/hh (per half hour) ']=master['KWH/hh (per half hour) '].astype('float')
grouped=master.groupby(['LCLid'])

nombres = {}


for user in range(len(masteruser)):
    usergroup=grouped.get_group(masteruser[user])
    groupdates=usergroup.Date.unique().tolist()
    usergrdate=usergroup.groupby(['Date'])
    old=usergrdate.get_group(groupdates[0])
    for numberdays in range(len(groupdates)):
        current= usergrdate.get_group(groupdates[numberdays])       
        demandold=old.loc[old['KWH/hh (per half hour) '].idxmax()]
        maxold=demandold['KWH/hh (per half hour) ']
        demandcurrent=current.loc[current['KWH/hh (per half hour) '].idxmax()]        
        maxcurrent=demandcurrent['KWH/hh (per half hour) ']
        if maxcurrent>maxold:
            old=current
        else: 
            continue       
    masterfilt=masterfilt.append(old)

for ind in range(len(masteruser)):
    auxlist=[]
    for halfhour in range(47):
        df4 = masterfilt[masterfilt['Hour']==horas[halfhour]] #filter at 12:30 am this has to be iterative to find all hours in the day
        df4 = df4[df4['LCLid']==masteruser[ind]]
        vals=df4['KWH/hh (per half hour) '].sum()
        #indivusertype=df4['Acorn'].iloc[0]
        auxlist.append(vals)
    #auxlist.append(indivusertype)
    nombres[masteruser[ind]] = auxlist
           
data = pd.DataFrame(nombres)

# the linkage function may receive custom distances in a compact vector format
comb=len(username)

distance_vector = np.zeros((1,len(username)*(len(username)-1)/2))
a = 0
for d1 in range(len(username)-1):
    for d2 in range(d1+1,len(username)):
        distance_vector[0][a] = DTW(data.iloc[:,d1], data.iloc[:,d2])
        a += 1

n_clust = 5
Z=linkage(distance_vector[0], method='ward')
dendrogram(Z, truncate_mode='lastp', p=n_clust, get_leaves=True)
clusters = fcluster(Z, n_clust, criterion='maxclust')
clusters_len = np.zeros((n_clust,1))
rep_curves = np.zeros((n_clust,47))
for c in range(n_clust):
    for d in range(len(username)):
        if clusters[d] == c+1:
            clusters_len[c] += 1
            for j in range(47):
                rep_curves[c][j] += data.iloc[j,d]
    rep_curves[c] /= clusters_len[c]
    
for i in rep_curves:
    plt.plot(i)