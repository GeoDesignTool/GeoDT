# ****************************************************************************
#### GeoDT Bulk Visualization
# ****************************************************************************

# ****************************************************************************
#### standard imports
# ****************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import pylab
import copy
from sklearn.linear_model import LinearRegression
import scipy.stats
import math
import os
import ipywidgets
# import ipympl
import GeoDT as gt
import pandas as pd
deg = gt.deg
MPa = gt.MPa
GPa = gt.GPa
yr = gt.yr
cP = gt.cP
mD = gt.mD
nbin = 50

# *********************************************************************
#### each vs each vs Guggliemi
# *********************************************************************
def hist2d(x=[],y=[],bins=[],log=False,heat=True,x_label='',y_label=''):
    # *********************************************************************
    # bins
    if not(bins):
        bins = [30,30]
    # 2d histogram
    x_val = []
    y_val = []
    if log:
        x_val = np.log10(x)
        y_val = np.log10(y)
    else:
        x_val = x
        y_val = y
    # replace nans with small number
    np.nan_to_num(x_val,copy=False,nan=1e-10)
    np.nan_to_num(y_val,copy=False,nan=1e-10)
    H, xedges, yedges = np.histogram2d(x_val,y_val,bins=bins,density=False)
    fig = pylab.figure(figsize=(10.3,5),dpi=100)
    ax1 = fig.add_subplot(111)
    # heatmap
    if heat:
        ax1.imshow(np.transpose(H[:,:]),aspect='auto',cmap='hot',extent=(xedges[0],xedges[-1],yedges[0],yedges[-1]),origin='lower',zorder=-1)
    # CDF
    csum = np.zeros((len(xedges)-1,len(yedges)-1),dtype=float)
    for j in range(1,len(yedges)):
        csum[:,j-1] = np.sum(H[:,:j],axis=1)
    for i in range(1,len(xedges)):
        if np.sum(H[i-1,:]) > 0:
            csum[i-1,:] = csum[i-1,:]/np.sum(H[i-1,:])
    # interpolate from CDF to get confidence intervals
    cis = np.asarray([0.05,0.50,0.95])
    vis = np.zeros((len(cis),len(xedges)-1),dtype=float)
    for i in range(0,len(xedges)-1):
        vis[:,i] = np.interp(cis,csum[i,:],yedges[1:])
    # plot 50% line
    if log:
        ax1.plot(10.0**xedges[1:],10.0**vis[1,:],'-',color='grey')
    else:
        ax1.plot(xedges[1:],vis[1,:],'-',color='grey')
    # plot confidence interval
    if log:
        ax1.fill_between(10.0**xedges[1:],10.0**vis[0,:],10.0**vis[2,:],color='grey', alpha=0.5)
    else:
        ax1.fill_between(xedges[1:],vis[0,:],vis[2,:],color='grey', alpha=0.5)
    # plot data
    if not(heat):
        ax1.plot(x_val,y_val,'.',label='results',color='grey')
    # ax1.plot(x_val,y_val,'.',label='results',color='grey')
    
    # formatting
    ax1.legend(loc='upper left', fancybox=True, prop={'size':8}, shadow=False, ncol=1, numpoints=1)
    ax1.set_xlabel(x_label,fontsize=10)
    ax1.set_ylabel(y_label,fontsize=10)
    if log:
        ax1.set_yscale('log')
        ax1.set_xscale('log')
    plt.tight_layout()
    
# ****************************************************************************
#### main program
# ****************************************************************************
#import data
filename = 'inputs_results_random.txt'
data = np.recfromcsv(filename,delimiter=',',filling_values=np.nan,deletechars='()',case_sensitive=True,names=True)

#restructure timeseries data
names = data.dtype.names
LifeSpan = data['LifeSpan'][0]/yr
TimeSteps = int(data['TimeSteps'][0])
t = np.linspace(0,LifeSpan,TimeSteps+1)[:-1]
hpro = np.zeros((len(data),TimeSteps),dtype=float)
Pout = np.zeros((len(data),TimeSteps),dtype=float)
dhout = np.zeros((len(data),TimeSteps),dtype=float)
j = [0,0,0]
for i in range(0,len(names)):
    if 'hpro' in names[i]:
        hpro[:,j[0]] = data[names[i]]
        j[0] += 1
    elif 'Pout' in names[i]:
        Pout[:,j[1]] = data[names[i]]
        j[1] += 1
    elif 'dhout' in names[i]:
        dhout[:,j[2]] = data[names[i]]
        j[2] += 1

# #enthalpy output histograms
# x_lab = 'BH_T'
# y_bin = np.linspace(-0.1e6,1.0e6,nbin+1)
# x_bin = np.linspace(np.min(data[x_lab]),np.max(data[x_lab]),int(0.5*nbin)+1)
# hist2d(x=data[x_lab],y=dhout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='reservoir temp (K)',y_label='thermal power output (kJ/s)')

# x_lab = 'Qinj'
# y_bin = np.linspace(-0.1e6,1.0e6,nbin+1)
# x_bin = np.linspace(0.0001,0.1,int(0.5*nbin)+1)
# hist2d(x=data[x_lab],y=dhout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='interval injection rate (m3/s)',y_label='thermal power output (kJ/s)')

# x_lab = 'w_spacing'
# y_bin = np.linspace(-0.1e6,1.0e6,nbin+1)
# x_bin = np.linspace(np.min(data[x_lab]),np.max(data[x_lab]),int(0.5*nbin)+1)
# hist2d(x=data[x_lab],y=dhout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='well spacing (m)',y_label='thermal power output (kJ/s)')

# x_lab = 'w_count'
# y_bin = np.linspace(-0.1e6,1.0e6,nbin+1)
# x_bin = np.linspace(0.5,np.max(data[x_lab])+0.5,int(np.max(data[x_lab])+1.2))
# hist2d(x=data[x_lab],y=dhout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='number of production wells',y_label='thermal power output (kJ/s)')

# x_lab = 'w_intervals'
# y_bin = np.linspace(-0.1e6,1.0e6,nbin+1)
# x_bin = np.linspace(0.5,np.max(data[x_lab])+0.5,int(np.max(data[x_lab])+1.2))
# hist2d(x=data[x_lab],y=dhout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='number of injection intervals',y_label='thermal power output (kJ/s)')

#power output histograms
x_lab = 'BH_T'
y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
x_bin = np.linspace(np.min(data[x_lab]),np.max(data[x_lab]),int(0.5*nbin)+1)
hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='reservoir temp (K)',y_label='electrical power output (kW)')

x_lab = 'Qinj'
y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
x_bin = np.linspace(0.0001,0.1,int(0.5*nbin)+1)
hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='interval injection rate (m3/s)',y_label='electrical power output (kW)')

x_lab = 'w_spacing'
y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
x_bin = np.linspace(np.min(data[x_lab]),np.max(data[x_lab]),int(0.5*nbin)+1)
hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='well spacing (m)',y_label='electrical power output (kW)')

x_lab = 'w_count'
y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
x_bin = np.linspace(0.5,np.max(data[x_lab])+0.5,int(np.max(data[x_lab])+1.2))
hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='number of production wells',y_label='electrical power output (kW)')

x_lab = 'w_intervals'
y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
x_bin = np.linspace(0.5,np.max(data[x_lab])+0.5,int(np.max(data[x_lab])+1.2))
hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='number of injection intervals',y_label='electrical power output (kW)')

# ****************************************************************************
#### top % boxplot normalized
# ****************************************************************************
#filter to top %
p = 0.10
rank = np.argsort(Pout[:,-1])[-int(p*len(Pout)):]
keep = np.zeros(len(Pout),dtype=bool)
for i in rank:
    keep[i] = True
top = data[keep]

#normalize ranges for every category
for n in names:
    top[n] = (top[n]-np.min(top[n]))/(np.max(top[n])-np.min(top[n]))

#boxplot
df = pd.DataFrame(top)
boxplot = df.boxplot(column=names)
# fig = pylab.figure(figsize=(10.3,5),dpi=100)
# ax1 = fig.add_subplot(111)
# plt.boxplot(top[n])





# ****************************************************************************
#### bottom % boxplot normalized
# ****************************************************************************
p = 0.10
rank = np.argsort(Pout[:,-1])[:int(p*len(Pout))]
keep = np.zeros(len(Pout),dtype=bool)
for i in rank:
    keep[i] = True
bot = data[keep]

pylab.show()

