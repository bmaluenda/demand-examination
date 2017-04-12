# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:54:48 2017

@author: inodu_desk_01
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#################
#   EDIT THIS   #
excluded_day_indices = [86,248]

#################

df = pd.read_excel("C:/Users/inodu_desk_01/Desktop/Consumo Cristalerias 2014.xlsx")
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

##### Processing data for monthly demand #####


#find max during "peak months and peak hours"
Dmax=[np.amax(data[[d for d in range(n_days) if date(2014,1,1) + timedelta(days=d) >= date(2014,m+1,1) and date(2014,1,1) + timedelta(days=d) <= date(2014,m+1,days_in_month[m])],:]) for m in range(12)]
Dmin=[np.amin(data[[d for d in range(n_days) if date(2014,1,1) + timedelta(days=d) >= date(2014,m+1,1) and date(2014,1,1) + timedelta(days=d) <= date(2014,m+1,days_in_month[m])],:]) for m in range(12)]

Dmaxpeak=[np.amax(data[[d for d in range(n_days) if date(2014,1,1) + timedelta(days=d) >= date(2014,m+1,1) and date(2014,1,1) + timedelta(days=d) <= date(2014,m+1,days_in_month[m])],:][:,[h for h in range(18,23)]]) for m in range(12)]
Dmaxnotpeak=[np.amax(data[[d for d in range(n_days) if date(2014,1,1) + timedelta(days=d) >= date(2014,m+1,1) and date(2014,1,1) + timedelta(days=d) <= date(2014,m+1,days_in_month[m])],:][:,[h for h in range(0,18)]+[23]]) for m in range(12)]

demandinfo={'Dmax': Dmax, 'Dmaxpeak': Dmaxpeak, 'Dmaxnotpeak': Dmaxnotpeak, 'Dmin':Dmin}
table=pd.DataFrame(demandinfo, index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']).astype(float)

table.to_csv('Load_Demand_Info.csv')
