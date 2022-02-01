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
    if not(heat):
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
filename = 'inputs_results_collabE2.csv'
data = np.recfromcsv(filename,delimiter=',',filling_values=np.nan,deletechars='()',case_sensitive=True,names=True)

#prefilter
if False:
    keep = np.zeros(len(data),dtype=bool)
    criteria = [['ResDepth',6000.0,10000.0],
                ['Qinj',0.015,0.050],
                ['w_spacing',500.0,800.0],
                ['w_count',3,5]]
    for i in range(0,len(criteria)):
        data = data[data[criteria[i][0]] > criteria[i][1]]
        data = data[data[criteria[i][0]] < criteria[i][2]]

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

#placeholder for electrical energy generation with binary cycle
Eout = np.asarray(Pout)
np.nan_to_num(Eout,copy=False,nan=0.0)

#identify result categories
#producing (P = producting, N = non-producing, M = marginal)
cat_pro = np.full(len(data),'M')
cat_pro[Eout[:,-1] > 5.0e3] = 'P'
cat_pro[Eout[:,-1] < 0.1e3] = 'N'

#caging (C = caged, L = leaky, S = seismic, O = other)
cat_cag = np.full(len(data),'C')
np.nan_to_num(data['qinj'],copy=False,nan=0.00000001)
np.nan_to_num(data['qpro'],copy=False,nan=0.000000001)
data['qinj'][data['qinj'] < 0.00000001] = 0.00000001
cat_cag[(-data['qpro']/data['qinj']) < 0.8] = 'O'
cat_cag[(-data['qpro']/data['qinj']) > 1.25] = 'O'
cat_cag[(-data['qpro']/data['qinj']) < 0.4] = 'L'
cat_cag[(-data['qpro']/data['qinj']) > 2.5] = 'L'
cat_cag[(data['max_quake'] > 3.0)] = 'S'

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
#### normalize data ranges
# ****************************************************************************
#normalize ranges
data_n = copy.deepcopy(data)
for n in names:
    np.nan_to_num(data_n[n],copy=False,nan=0.0)
    if n == 'w_skew':
        data_n[n] = np.abs(data_n[n])
    span = np.max(data_n[n])-np.min(data_n[n]) 
    if span > 0:
        data_n[n] = (data_n[n]-np.min(data_n[n]))/span
    else:
        data_n[n] = 0.0

# ****************************************************************************
#### caged-productive boxplot normalized
# ****************************************************************************
#data columns of interest
column=['ResDepth','ResGradient','Qinj','w_spacing','w_count','w_length','w_azimuth','w_dip','w_proportion','w_phase','w_toe','w_skew','w_intervals','ra']

#filter to caged & productive categories
keep = np.zeros(len(data),dtype=bool)
for i in range(0,len(data)):
    if (cat_pro[i] == 'P') and cat_cag[i] == 'C':
        keep[i] = True
data_PC = data_n[keep]

#boxplot
fig = pylab.figure(figsize=(16,10),dpi=100)
ax1 = fig.add_subplot(411)
df = pd.DataFrame(data_PC,columns=names)
boxplot = df.boxplot(grid=False,column=column)
ax1.set_ylabel('A: caged-productive')

# ****************************************************************************
#### leaky-productive boxplot normalized
# ****************************************************************************
#filter to leaky & productive categories
keep = np.zeros(len(data),dtype=bool)
for i in range(0,len(data)):
    if (cat_pro[i] == 'P') and cat_cag[i] == 'L':
        keep[i] = True
data_PL = data_n[keep]

#boxplot
ax2 = fig.add_subplot(412)
df = pd.DataFrame(data_PL,columns=names)
boxplot = df.boxplot(grid=False,column=column)
ax2.set_ylabel('B: leaky-productive')

# ****************************************************************************
#### seismic boxplot normalized
# ****************************************************************************
#filter to seismogenic categories
keep = np.zeros(len(data),dtype=bool)
for i in range(0,len(data)):
    if cat_cag[i] == 'S':
        keep[i] = True
data_S = data_n[keep]

#boxplot
ax3 = fig.add_subplot(413)
df = pd.DataFrame(data_S,columns=names)
boxplot = df.boxplot(grid=False,column=column)
ax3.set_ylabel('C: seismogenic')

# ****************************************************************************
#### leaky-nonproductive boxplot normalized
# ****************************************************************************
#filter to leaky & non-productive categories
keep = np.zeros(len(data),dtype=bool)
for i in range(0,len(data)):
    if (cat_pro[i] == 'N') and cat_cag[i] == 'L':
        keep[i] = True
data_NL = data_n[keep]

#boxplot
ax4 = fig.add_subplot(414)
df = pd.DataFrame(data_NL,columns=names)
boxplot = df.boxplot(grid=False,column=column)
ax4.set_ylabel('F: leaky non-productive')

# ****************************************************************************
#### formatting
# ****************************************************************************
plt.tight_layout()

# ****************************************************************************
#### pie charts
# ****************************************************************************
#float length
count = float(len(data))

# all
labels = ['A: caged-productive','B: leaky-productive','C: seismogenic','F: leaky-nonproductive','O: Other']
sizes = [len(data_PC)/count,len(data_PL)/count,len(data_S)/count,len(data_NL)/count]
sizes += [1.0 - np.sum(sizes)]
explode = (0.1,0,0,0,0)
fig = pylab.figure(figsize=(5,5),dpi=100)
ax1 = fig.add_subplot(111)
ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
ax1.axis('equal')
ax1.set_title('dataset from %i samples' %(count))
print('\n*** PIE #1')
for n in column:
    print([n,np.min(data[n]),np.average(data[n]),np.max(data[n])])
    
#format
plt.tight_layout()





# # ****************************************************************************
# #### bottom % boxplot normalized
# # ****************************************************************************
# p = 0.10
# rank = np.argsort(Pout[:,-1])[:int(p*len(Pout))]
# keep = np.zeros(len(Pout),dtype=bool)
# for i in rank:
#     keep[i] = True
# bot = data[keep]

pylab.show()

