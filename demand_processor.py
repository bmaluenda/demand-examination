import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

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