import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#################
#   EDIT THIS   #
excluded_day_indices = [86,248]
n_clust = 4
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

pdf_plotter = PdfPages('outputs.pdf')
def fig_ax(title, ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim([0,ylim*1.07])
        ax.set_yticks(np.arange(0, ylim*1.07, 1000))
    return fig, ax
def save_close(fig, extra=None):
    if extra is None:
        pdf_plotter.savefig(fig)
    else:
        pdf_plotter.savefig(fig, bbox_extra_artists=(extra,), bbox_inches='tight')
    plt.close()
    pass

df = pd.read_excel("C:/Users/inodu_notebook_01/Desktop/Consumo Cristalerias 2014.xlsx")
fig,ax = fig_ax('Consumo Cristalerias Chile 2014 (kWh)', df.max(0)[0])
df.plot(ax=ax, use_index=True, legend=False, linewidth=0.5)
save_close(fig)

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
max_value = np.amax(data)

# the linkage function may receive custom distances in a compact vector format
distance_vector = np.zeros((1,n_days*(n_days-1)/2))
a = 0
for d1 in range(n_days-1):
    for d2 in range(d1+1,n_days):
        distance_vector[0][a] = DTW(data[d1], data[d2])
        a += 1

Z=linkage(distance_vector[0], method='ward')
fig, ax = fig_ax('Dendrogram')
dendrogram(Z, truncate_mode='lastp', p=n_clust, get_leaves=True)
save_close(fig)
clusters = fcluster(Z, n_clust, criterion='maxclust')

# Representative curves for each cluster
clusters_len = np.zeros((n_clust,1))
rep_curves = np.zeros((n_clust,24))
fig, ax = fig_ax('Average demand curves for each cluster (kW)', max_value)
for c in range(n_clust):
    for d in range(n_days):
        if clusters[d] == c+1:
            clusters_len[c] += 1
            for j in range(24):
                rep_curves[c][j] += data[d][j]
    rep_curves[c] /= clusters_len[c]
    ax.plot(rep_curves[c], linewidth=0.02*clusters_len[c], \
             label='Cluster No. '+str(c+1)+'. Groups '+str(int(clusters_len[c][0]))+' days.')
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
save_close(fig, lgd)

# Curves in each cluster
for c in range(n_clust):
    fig, ax = fig_ax('Daily sampled and average demand curves (kW) \n Cluster No. '+str(c+1), max_value)
    for d_ind, c_ind in enumerate(clusters):
        if c_ind == c+1:
            plt.plot(data[d_ind],color='blue')
    ax.plot(rep_curves[c],color='red')
    save_close(fig)

# State of operation (cluster number)
fig, ax = fig_ax('State of operation (cluster number) by day')
ax.set_yticks(np.arange(1, n_clust+1, 1))
ax.plot([i for i in range(n_days)],clusters,".")
save_close(fig)

for c in range(n_clust):
    # Demand distributions in each hour for each operational mode
    fig, ax = fig_ax('Demand distributions in each hour for each operational mode \n Cluster No. '+str(c+1), max_value)
    ax.boxplot(data[[d for d in range(n_days) if clusters[d] == c+1],:])
    save_close(fig)
    # Histogram of demand distribution for all hours in each operational mode
    fig, ax = fig_ax('Histogram of demand distribution for all hours in each operational mode \n Cluster No. '+str(c+1))
    ax.set_xlim([0,max_value*1.07])
    ax.hist(data[[d for d in range(n_days) \
                   if clusters[d] == c+1],:].flatten(), \
                        normed=False, bins='auto')
    save_close(fig)
fig, ax = fig_ax('Histogram of demand distribution for all hours and days')
ax.set_xlim([0,max_value*1.07])
ax.hist(data.flatten(), normed=False, bins=300)
save_close(fig)
pdf_plotter.close()