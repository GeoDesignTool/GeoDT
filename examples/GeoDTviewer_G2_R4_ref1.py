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
from iapws import IAPWS97 as therm
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

# *********************************************************************
#### 2D histogram with probability ratios
# *********************************************************************
def prob2drcat(x=[],cat=[],nbin=50,names=[],labels=[],x_label='',y_label=''):
    # histogram bins
    bins = np.linspace(np.min(x),np.max(x),nbin+1)
    
    # identify bin index for each x value
    ibin = np.asarray((nbin)*(x-np.min(x))/(np.max(x)-np.min(x)))
    ibin = ibin.astype(int)
    
    # probability matrix
    prob = np.zeros((nbin+1,len(names)),dtype=float)
    for i in range(0,len(cat)):
        for n in range(0,len(names)):
            if cat[i] == names[n]:
                prob[ibin[i],n] += 1
                
    # lump max bin into second from top bin and remove max bin
    prob[-2,:] += prob[-1,:]
    prob = prob[:-1,:]
    
    # normalize to 0 to 1
    for b in range(0,nbin):
        if np.sum(prob[b,:]) > 0:
            prob[b,:] = prob[b,:]/np.sum(prob[b,:])
    
    # color plots
    colors = ['blue','orange','green','red']
    fig = pylab.figure(figsize=(10.3,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.fill_between(bins[1:],np.zeros(nbin),prob[:,0],color=colors[0], alpha=0.5, label=labels[0])
    for n in range(1,len(names)):
        ax1.fill_between(bins[1:],np.sum(prob[:,:n],axis=1),prob[:,n] + np.sum(prob[:,:n],axis=1),color=colors[n], alpha=0.5, label=labels[n])

    # formatting
    ax1.legend(loc='upper left', fancybox=True, prop={'size':8}, shadow=False, ncol=1, numpoints=1)
    ax1.set_xlabel(x_label,fontsize=10)
    ax1.set_ylabel(y_label,fontsize=10)
    # ax1.set_ylim(0.0,1.0)
    plt.tight_layout()
    
# *********************************************************************
#### enthalpy function for this system
# *********************************************************************
def enthalpy(T_lo_K,T_hi_K,P_nom_MPa,plot=True):
    #****** enthalpy function linearization *******
    x = np.linspace(T_lo_K,T_hi_K,100,dtype=float)
    y = np.zeros(100,dtype=float)
    for i in range(0,len(x)):
        y[i] = therm(T=x[i],P=P_nom_MPa).h #kJ/kg
    T_to_h = np.polyfit(x,y,3)
    h_to_T = np.polyfit(y,x,3)
    
    #plotting
    if plot:
        y2 = T_to_h[0]*x**3.0 + T_to_h[1]*x**2.0 + T_to_h[2]*x**1.0 + T_to_h[3]
        x2 = h_to_T[0]*y**3.0 + h_to_T[1]*y**2.0 + h_to_T[2]*y**1.0 + h_to_T[3]
        fig = pylab.figure(figsize=(8.0, 6.0), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
        ax1 = fig.add_subplot(111)
        ax1.plot(x,y,label='raw')
        ax1.plot(x,y2,label='fitted')
        ax1.plot(x2,y,label='reverse')
        ax1.set_ylabel('Enthalpy (kJ/kg)')
        ax1.set_xlabel('Temperature (K)')
        ax1.legend(loc='upper left', prop={'size':8}, ncol=2, numpoints=1)
    
    return T_to_h, h_to_T 

# ****************************************************************************
#### main program
# ****************************************************************************
#import data
filename = 'inputs_results_random.csv'
data = np.recfromcsv(filename,delimiter=',',filling_values=np.nan,deletechars='()',case_sensitive=True,names=True)
names = data.dtype.names
# for n in names:
#     np.nan_to_num(data[n],copy=False,nan=0.0)

print('samples = %i' %(len(data)))

#prefilter
if False:
    keep = np.zeros(len(data),dtype=bool)
    criteria = [['ResDepth',7000,9400],
                ['ResGradient',45,55],
                ['Ks3',0.6,0.9],
                ['Qinj',0.032,0.042],
                ['w_count',2.9,6.1],
                ['w_azimuth',0.0*deg,45.0*deg],
                ['w_dip',0.0*deg,45.0*deg],
                ['w_proportion',0.56,0.88],
                ['w_intervals',5.8,12.1],
                ['ra',0.12,0.19]
                ]
    for i in range(0,len(criteria)):
        data = data[data[criteria[i][0]] > criteria[i][1]]
        data = data[data[criteria[i][0]] < criteria[i][2]]
# if False:
#     data['q13'][data['q13'] < 1.0e-7] = 1.0e-7
#     recovery2 = (-(data['q6']+data['q7']+data['q8']+data['q9'])/data['q13'])
#     data = data[(recovery2 > 0.8)*(recovery2 < 1.25)]
        
print('filtered samples = %i' %(len(data)))

#restructure timeseries data

LifeSpan = data['LifeSpan'][0]/yr
TimeSteps = int(data['TimeSteps'][0])
ts = np.linspace(0,LifeSpan,TimeSteps+1)[:-1]
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

#get fluid temperature change
np.nan_to_num(data['qinj'],copy=False,nan=1e-7)
T_lo = np.min([np.min(data['Tinj']),np.min(data['BH_T'])])
T_to_h, h_to_T = enthalpy(np.min(data['Tinj'])+273.15,np.max(data['BH_T']),np.min(data['BH_P'])*10**-6,True)
Tpro = np.zeros((len(data),TimeSteps),dtype=float)
for t in range(0,TimeSteps):
    Tpro[:,t] = h_to_T[0]*hpro[:,t]**3.0 + h_to_T[1]*hpro[:,t]**2.0 + h_to_T[2]*hpro[:,t] + h_to_T[3]

#placeholder for electrical energy generation with binary cycle
Eout = np.asarray(Pout)
np.nan_to_num(Eout,copy=False,nan=0.0)

#identify result categories
#producing (P = producting, N = non-producing, M = marginal)
cat_pro = np.full(len(data),'M')
cat_pro[Eout[:,-1] > 5.0e3] = 'P'
cat_pro[Eout[:,-1] < 0.1e3] = 'N'
cat_pro[Eout[:,-1] > 20.0e3] = 'S'

#caging (C = caged, L = losing, O = other. G = gaining)
cat_cag = np.full(len(data),'C')
np.nan_to_num(data['qinj'],copy=False,nan=1e-7)
np.nan_to_num(data['qpro'],copy=False,nan=1e-8)
data['qinj'][data['qinj'] < 1e-7] = 1e-7
recovery = -data['qpro']/data['qinj']
cat_cag[recovery < 0.8] = 'O'
cat_cag[recovery > 1.25] = 'O'
cat_cag[recovery < 0.4] = 'L'
cat_cag[recovery > 2.5] = 'G'

#seismic vs not
cat_seis = np.full(len(data),'I')
cat_seis[(data['max_quake'] > 1.0)] = 'D'
cat_seis[(data['max_quake'] > 3.0)] = 'A'
cat_seis[(data['max_quake'] > 5.0)] = 'R'

#shear vs tensile
cat_shea = np.full(len(data),'U')
cat_shea[(data['hfstim'] > 0.0)] = 'H'
cat_shea[(data['nfstim'] > 0.0)] = 'S'
cat_shea[(data['nfstim']*data['hfstim'] > 0.0)] = 'M'

#thermal breakthrough
ti = int(0.5*len(ts)) #int((2.0/12.0)/(LifeSpan/TimeSteps))+1
cat_ther = np.full(len(data),'N')
cat_ther[np.abs((data['BH_T']-Tpro[:,-1])) > 10.0] = 'E' #breakthrough before final timestep
cat_ther[np.abs((data['BH_T']-Tpro[:,ti])) > 10.0] = 'O' #intermediate breakthrough time
cat_ther[np.abs((data['BH_T']-Tpro[:,3])) > 10.0] = 'R' #breakthrough in first time step

# #enthalpy output histograms
# x_lab = 'w_spacing'
# y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
# x_bin = np.linspace(np.min(data[x_lab]),np.max(data[x_lab]),int(0.5*nbin)+1)
# hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='well spacing (m)',y_label='electrical power output (kW)')

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
# lohi = [-3.0e-4,3.0e-4]        
# for n in range(0,15):
#     na = 'q%i' %(n)
#     data_n[na] = (data[na]-lohi[0])/(lohi[1]-lohi[0])
#     data_n[na][data_n[na] > 1.0] = 1.0
#     data_n[na][data_n[na] < 0.0] = 0.0
    


if True:
    # ****************************************************************************
    #### BOXPLOTS!
    # ****************************************************************************
    #data columns of interest
    column=['ResDepth','ResGradient','ResE','Ks3','Qstim','Vstim','Qinj','w_count','w_spacing','w_length','w_azimuth','w_dip','w_proportion','w_phase','w_toe','w_skew','w_intervals','ra']
    
    #filter to caged & productive categories
    keep = np.zeros(len(data),dtype=bool)
    for i in range(0,len(data)):
        if ((cat_pro[i] == 'P') or (cat_pro[i] == 'S')) and (cat_cag[i] == 'C'):
            keep[i] = True
    data_PC = data_n[keep]
    fig = pylab.figure(figsize=(16,10),dpi=100)
    ax1 = fig.add_subplot(411)
    df = pd.DataFrame(data_PC,columns=names)
    boxplot = df.boxplot(grid=False,column=column)
    ax1.set_ylabel('A: caged >5 MWe (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
    #filter to star performers
    keep = np.zeros(len(data),dtype=bool)
    for i in range(0,len(data)):
        if (cat_pro[i] == 'S'):
            keep[i] = True
    data_PL = data_n[keep]
    ax2 = fig.add_subplot(412)
    df = pd.DataFrame(data_PL,columns=names)
    boxplot = df.boxplot(grid=False,column=column)
    ax2.set_ylabel('B: power >20 MWe (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
    #filter to seismogenic categories
    keep = np.zeros(len(data),dtype=bool)
    for i in range(0,len(data)):
        if (cat_seis[i] == 'A') or (cat_seis[i] == 'R'):
            keep[i] = True
    data_S = data_n[keep]
    ax3 = fig.add_subplot(413)
    df = pd.DataFrame(data_S,columns=names)
    boxplot = df.boxplot(grid=False,column=column)
    ax3.set_ylabel('C: seismic >3.0 Mw (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
    #filter to failed scenarios
    keep = np.zeros(len(data),dtype=bool)
    for i in range(0,len(data)):
        if (cat_pro[i] == 'N') and (cat_cag[i] == 'L'):
            keep[i] = True
    data_NL = data_n[keep]
    ax4 = fig.add_subplot(414)
    df = pd.DataFrame(data_NL,columns=names)
    boxplot = df.boxplot(grid=False,column=column)
    ax4.set_ylabel('F: leaky <0.1 MWe (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
    #formatting
    plt.tight_layout()
    
    #data ranges
    for n in column:
        print([n,np.min(data[n]),np.average(data[n]),np.max(data[n])])

    # ****************************************************************************
    #### PIE CHARTS!
    # ****************************************************************************
    #### pie chart: productive
    labels = ['P', 'S', 'M', 'N']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_pro[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['P = >5 MWe','S = >20 MWe', 'M = other', 'N = <0.1 MWe']
    explode = (0.1,0.1,0,0)
    fig = pylab.figure(figsize=(5,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    #format
    plt.tight_layout()
    plt.savefig('prod.png', format='png')
    
    #### pie chart: caged
    labels = ['C', 'O', 'G', 'L']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_cag[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['C = 80-125%','O = other', 'G = >250%', 'L = <40%']
    explode = (0.1,0,0,0)
    fig = pylab.figure(figsize=(5,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    #format
    plt.tight_layout()
    plt.savefig('cage.png', format='png')

    #### pie chart: breakthrough
    labels = ['R', 'O', 'E', 'N']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_ther[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['R = in %.2f yr' %(ts[3]), 'O = in %.2f yr' %(ts[ti]), 'E = in %.2f yr' %(ts[-1]), 'N = no breakthrough']
    explode = (0.1,0,0,0)
    fig = pylab.figure(figsize=(5,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    #format
    plt.tight_layout()
    plt.savefig('break.png', format='png')

    #### pie chart: seismic
    labels = ['I', 'D', 'A', 'R']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_seis[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['I = Innactive','D = >1.0 Mw','A = >3.0 Mw','R = >5.0 Mw']
    explode = (0.1,0,0,0)
    fig = pylab.figure(figsize=(5,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    #format
    plt.tight_layout()
    plt.savefig('seis.png', format='png')

    #### pie chart: shear
    labels = ['S', 'M', 'H', 'U']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_shea[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['S = Hydroshear','M = Mixed', 'H = Hydrofrac', 'U = None']
    explode = (0,0,0,0)
    fig = pylab.figure(figsize=(5,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    #format
    plt.tight_layout()
    plt.savefig('shear.png', format='png')
    
    # ****************************************************************************
    #### Probability Plots
    # ****************************************************************************
    # probability plot of production vs well spacing
    x = data['w_spacing']
    prob2drcat(x=x,
                cat=cat_pro,
                nbin=30,
                names=['S', 'P', 'M', 'N'],
                labels=['S = >20 MWe', 'P = >5 MWe', 'M = other', 'N = <0.1 MWe'],
                x_label= 'Well Spacing (m)',
                y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('w_spacing.png', format='png')
    
    # probability plot of production vs well count
    x = data['w_count']
    prob2drcat(x=x,
                cat=cat_pro,
                nbin=4,
                names=['S', 'P', 'M', 'N'],
                labels=['S = >20 MWe', 'P = >5 MWe', 'M = other', 'N = <0.1 MWe'],
                x_label= 'Well Count (producers)',
                y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('w_count.png', format='png')
    
    # probability plot of production vs injection rate
    x = data['Qinj']
    prob2drcat(x=x,
                cat=cat_pro,
                nbin=30,
                names=['S', 'P', 'M', 'N'],
                labels=['S = >20 MWe', 'P = >5 MWe', 'M = other', 'N = <0.1 MWe'],
                x_label= 'Circulation Injection Rate (m3/s)',
                y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('Qinj.png', format='png')
    
    # probability plot of production vs injection intervals
    x = data['w_intervals']
    prob2drcat(x=x,
                cat=cat_pro,
                nbin=8,
                names=['S', 'P', 'M', 'N'],
                labels=['S = >20 MWe', 'P = >5 MWe', 'M = other', 'N = <0.1 MWe'],
                x_label= 'Number of Injection Intervals (zones)',
                y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('w_intervals.png', format='png')
    
    # probability plot of production vs depth
    x = data['ResDepth']
    prob2drcat(x=x,
                cat=cat_pro,
                nbin=30,
                names=['S', 'P', 'M', 'N'],
                labels=['S = >20 MWe', 'P = >5 MWe', 'M = other', 'N = <0.1 MWe'],
                x_label= 'Reservoir Depth (m)',
                y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('ResDepth.png', format='png')

    # probability plot of seismicity vs well spacing
    x = data['w_spacing']
    prob2drcat(x=x,
                cat=cat_seis,
                nbin=30,
                names=['I', 'D', 'A', 'R'],
                labels=['I = innactive', 'D = >1.0 Mw', 'A = >3.0 Mw', 'R = >5.0 Mw'],
                x_label= 'Well Spacing (m)',
                y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('seis_spacing.png', format='png')
    
    # ****************************************************************************
    #### 2D HISTOGRAMS
    # ****************************************************************************
    # #adjust recovery
    # recovery[recovery > 2.5] = 2.5
    # recovery[recovery < 0.0] = 0.0
    
    # # probability plot of total recovery rate vs injection rate
    # hist2d(x=data['qinj'],y=recovery,bins=[50,50],log=False,heat=True,x_label='Injection Rate (m3/s)',y_label='Recovery Ratio (all wells)')
    # plt.tight_layout()
    # plt.savefig('recovery_byrate.png', format='png')
    
    # # probability plot of total recovery rate vs well count
    # hist2d(x=data['w_count'],y=recovery,bins=[int(np.max(data['w_count']+1)),50],log=False,heat=True,x_label='Well Count (producers)',y_label='Recovery Ratio (all wells)')
    # plt.tight_layout()
    # plt.savefig('recovery_bycount.png', format='png')
    
    # probability plot of max seismic moment vs well spacing
    hist2d(x=data['w_spacing'],y=data['max_quake'],bins=[50,50],log=False,heat=True,x_label='Well Spacing (m)',y_label='Max Quake (Mw)')
    plt.tight_layout()
    plt.savefig('seis_byspace.png', format='png')


# # ****************************************************************************
# #### hydroshear boxplot normalized
# # ****************************************************************************
# if True:
#     #data columns of interest
#     column = ['Qstim','Vstim','Qinj','inj_depth','dPp','BH_P','pinj','q0','q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','q13','q14']
    
#     # ****************************************************************************
#     #filter to caged - shear
#     keep = np.zeros(len(data),dtype=bool)
#     for i in range(0,len(data)):
#         if (cat_cag2[i] == 'C') and (cat_shea[i] == 'S'):
#             keep[i] = True
#     data_PC = data_n[keep]
#     #boxplot
#     fig = pylab.figure(figsize=(16,10),dpi=100)
#     ax1 = fig.add_subplot(411)
#     df = pd.DataFrame(data_PC,columns=names)
#     boxplot = df.boxplot(grid=False,column=column)
#     ax1.set_ylabel('A: caged-shear (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
#     # ****************************************************************************
#     #filter to contained - shear + mixed
#     keep = np.zeros(len(data),dtype=bool)
#     for i in range(0,len(data)):
#         if (cat_cag3[i] == 'C') and (cat_shea[i] == 'S'):
#             keep[i] = True
#         if (cat_cag3[i] == 'C') and (cat_shea[i] == 'M'):
#             keep[i] = True
#     data_PL = data_n[keep]
    
#     #boxplot
#     ax2 = fig.add_subplot(412)
#     df = pd.DataFrame(data_PL,columns=names)
#     boxplot = df.boxplot(grid=False,column=column)
#     ax2.set_ylabel('B: contained-mixed (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
#     # ****************************************************************************
#     #filter to breakthrough
#     keep = np.zeros(len(data),dtype=bool)
#     for i in range(0,len(data)):
#         if cat_ther[i] != 'N':
#             keep[i] = True
#     data_S = data_n[keep]
    
#     #boxplot
#     ax3 = fig.add_subplot(413)
#     df = pd.DataFrame(data_S,columns=names)
#     boxplot = df.boxplot(grid=False,column=column)
#     ax3.set_ylabel('C: breakthrough (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
#     # ****************************************************************************
#     #filter to uncontained hydrofrac or seismogenic
#     keep = np.zeros(len(data),dtype=bool)
#     for i in range(0,len(data)):
#         if (cat_shea[i] == 'H') and (cat_cag3[i] == 'L'):
#             keep[i] = True
#         if (cat_shea[i] == 'H') and (cat_cag3[i] == 'G'):
#             keep[i] = True
#         if (cat_seis[i] == 'A') or (cat_seis[i] == 'R'):
#             keep[i] = True
#     data_NL = data_n[keep]
    
#     #boxplot
#     ax4 = fig.add_subplot(414)
#     df = pd.DataFrame(data_NL,columns=names)
#     boxplot = df.boxplot(grid=False,column=column)
#     ax4.set_ylabel('F: leaky-hf or seismic (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
#     # ****************************************************************************
#     #### formatting
#     # ****************************************************************************
#     plt.tight_layout()
    
#     #print value ranges
#     print('\n*** PIE #1')
#     for n in column:
#         print([n,np.min(data[n]),np.average(data[n]),np.max(data[n])])
    
#     # ****************************************************************************
#     #### pie chart: caged
#     # ****************************************************************************
#     #float length
#     count = float(len(data))
    
#     # caged vs other
#     labels = ['C', 'O', 'L', 'G']
#     sizes = []
#     for i in range(0,len(labels)):
#         keep = np.zeros(len(data),dtype=int)
#         for j in range(0,len(data)):
#             if cat_cag2[j] == labels[i]:
#                 keep[j] = 1
#         sizes += [np.sum(keep)/len(keep)]
#     labels = ['C = caged','O = other', 'L = leak out', 'G = leak in']
#     explode = (0.1,0,0,0)
#     fig = pylab.figure(figsize=(5,5),dpi=100)
#     ax1 = fig.add_subplot(111)
#     ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
#     ax1.axis('equal')
#     # ax1.set_title('dataset from %i samples' %(count))
    
        
#     #format
#     plt.tight_layout()
#     plt.savefig('cage.png', format='png')

#     # ****************************************************************************
#     #### pie chart: contained
#     # ****************************************************************************
#     #float length
#     count = float(len(data))
    
#     # caged vs other
#     labels = ['C', 'O', 'L', 'G']
#     sizes = []
#     for i in range(0,len(labels)):
#         keep = np.zeros(len(data),dtype=int)
#         for j in range(0,len(data)):
#             if cat_cag3[j] == labels[i]:
#                 keep[j] = 1
#         sizes += [np.sum(keep)/len(keep)]
#     labels = ['C = contained','O = other', 'L = leak out', 'G = leak in']
#     explode = (0.1,0,0,0)
#     fig = pylab.figure(figsize=(5,5),dpi=100)
#     ax1 = fig.add_subplot(111)
#     ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
#     ax1.axis('equal')
#     # ax1.set_title('dataset from %i samples' %(count))
        
#     #format
#     plt.tight_layout()
#     plt.savefig('contain.png', format='png')

#     # ****************************************************************************
#     #### pie chart: breakthrough
#     # ****************************************************************************
#     #float length
#     count = float(len(data))
    
#     # caged vs other
#     labels = ['R', 'O', 'E', 'N']
#     sizes = []
#     for i in range(0,len(labels)):
#         keep = np.zeros(len(data),dtype=int)
#         for j in range(0,len(data)):
#             if cat_ther[j] == labels[i]:
#                 keep[j] = 1
#         sizes += [np.sum(keep)/len(keep)]
#     labels = ['R = in %.2f yr' %(ts[3]), 'O = in %.2f yr' %(ts[ti]), 'E = in %.2f yr' %(ts[-1]), 'N = no breakthrough']
#     explode = (0.1,0,0,0)
#     fig = pylab.figure(figsize=(5,5),dpi=100)
#     ax1 = fig.add_subplot(111)
#     ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
#     ax1.axis('equal')
#     # ax1.set_title('dataset from %i samples' %(count))
        
#     #format
#     plt.tight_layout()
#     plt.savefig('break.png', format='png')

#     # ****************************************************************************
#     #### pie chart: seismic
#     # ****************************************************************************
#     #float length
#     count = float(len(data))
    
#     # caged vs other
#     labels = ['I', 'R', 'A']
#     sizes = []
#     for i in range(0,len(labels)):
#         keep = np.zeros(len(data),dtype=int)
#         for j in range(0,len(data)):
#             if cat_seis[j] == labels[i]:
#                 keep[j] = 1
#         sizes += [np.sum(keep)/len(keep)]
#     labels = ['I = Mwmax < 1.0','R = Mwmax > 1.0','A = Mwmax > 3.0']
#     explode = (0.1,0,0)
#     fig = pylab.figure(figsize=(5,5),dpi=100)
#     ax1 = fig.add_subplot(111)
#     ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
#     ax1.axis('equal')
#     # ax1.set_title('dataset from %i samples' %(count))
        
#     #format
#     plt.tight_layout()
#     plt.savefig('seis.png', format='png')

#     # ****************************************************************************
#     #### pie chart: shear
#     # ****************************************************************************
#     #float length
#     count = float(len(data))
    
#     # caged vs other
#     labels = ['S', 'M', 'H', 'U']
#     sizes = []
#     for i in range(0,len(labels)):
#         keep = np.zeros(len(data),dtype=int)
#         for j in range(0,len(data)):
#             if cat_shea[j] == labels[i]:
#                 keep[j] = 1
#         sizes += [np.sum(keep)/len(keep)]
#     labels = ['S = Hydroshear','M = Mixed', 'H = Hydrofrac', 'U = None']
#     explode = (0.1,0,0,0)
#     fig = pylab.figure(figsize=(5,5),dpi=100)
#     ax1 = fig.add_subplot(111)
#     ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
#     ax1.axis('equal')
#     # ax1.set_title('dataset from %i samples' %(count))
        
#     #format
#     plt.tight_layout()
#     plt.savefig('shear.png', format='png')
    
# # color probability plots
# if True:
#     # probability plot of shear vs depth
#     x = data['inj_depth']*77.1
#     prob2drcat(x=x,
#                cat=cat_shea,
#                nbin=int((np.max(x)-np.min(x))/0.65)+1,
#                names=['S','M','H','U'],
#                labels= ['S = Hydroshear','M = Mixed', 'H = Hydrofrac', 'U = None'],
#                x_label= 'Injecton Depth in E2-TC (m)',
#                y_label= 'Ratios by Category')
#     plt.tight_layout()
#     plt.savefig('probshear.png', format='png')
    
#     # probability plot of shear vs depth
#     x = (data['BH_P']+data['dPp'])/MPa #m3/s
#     prob2drcat(x=x,
#                cat=cat_shea,
#                nbin=int((np.max(x)-np.min(x))/0.65)+1,
#                names=['S','M','H','U'],
#                labels= ['S = Hydroshear','M = Mixed', 'H = Hydrofrac', 'U = None'],
#                x_label= 'Production Pressure (MPa)',
#                y_label= 'Ratios by Category')
#     plt.tight_layout()
#     plt.savefig('probshearback.png', format='png')
    
#     # probability plot of breakthrough vs production pressure
#     x = (data['BH_P']+data['dPp'])/MPa #m3/s
#     prob2drcat(x=x,
#                cat=cat_ther,
#                nbin=50,
#                names = ['R', 'O', 'E', 'N'],
#                labels = ['R = in %.2f yr' %(ts[3]), 'O = in %.2f yr' %(ts[ti]), 'E = in %.2f yr' %(ts[-1]), 'N = no breakthrough'],
#                x_label= 'Production Pressure (MPa)',
#                y_label= 'Ratios by Category')
#     plt.tight_layout()
#     plt.savefig('probbreakback.png', format='png')
    
#     # probability plot of breakthrough vs injection rate
#     x = data['q13']/1.66667e-8 #m3/s
#     prob2drcat(x=x,
#                cat=cat_ther,
#                nbin=50,
#                names = ['R', 'O', 'E', 'N'],
#                labels = ['R = in %.2f yr' %(ts[3]), 'O = in %.2f yr' %(ts[ti]), 'E = in %.2f yr' %(ts[-1]), 'N = no breakthrough'],
#                x_label= 'Circulation Injecton Rate (mL/min)',
#                y_label= 'Ratios by Category')
#     plt.tight_layout()
#     plt.savefig('probbreak.png', format='png')
    
#     # probability plot of total recovery rate vs injection rate
#     recovery3[recovery3>2.5] = 2.5
#     recovery3[recovery3<0.4] = 0.4
#     hist2d(x=data['q13']/1.66667e-8,y=recovery3,bins=[50,50],log=False,heat=True,x_label='Injection Rate (mL/min)',y_label='Recovery Ratio (all wells)')
#     plt.tight_layout()
#     plt.savefig('probrecovery3.png', format='png')
    
#     # probability plot of caged recovery rate vs injection rate
#     recovery2[recovery2>2.5] = 2.5
#     recovery2[recovery2<0.4] = 0.4
#     hist2d(x=data['q13']/1.66667e-8,y=recovery2,bins=[50,50],log=False,heat=True,x_label='Injection Rate (mL/min)',y_label='Recovery Ratio (caged)')
#     plt.tight_layout()
#     plt.savefig('probrecovery2.png', format='png')
    
#     # probability plot of breakthrough vs peak production rate
#     x = np.max(-1.0*np.asarray([data['q0'],data['q1'],data['q2'],data['q3'],data['q4'],data['q5'],
#                                  data['q6'],data['q7'],data['q8'],data['q9'],
#                                  data['q12'],data['q14']]),axis=0)/1.66667e-8
#     prob2drcat(x=x,
#                cat=cat_ther,
#                nbin=200,
#                names = ['R', 'O', 'E', 'N'],
#                labels = ['R = in %.2f yr' %(ts[3]), 'O = in %.2f yr' %(ts[ti]), 'E = in %.2f yr' %(ts[-1]), 'N = no breakthrough'],
#                x_label= 'Peak Overall Production Rate (mL/min)',
#                y_label= 'Ratios by Category')
#     plt.tight_layout()
#     plt.savefig('probbreakpro.png', format='png')
    
#     # probability plot of breakthrough vs peak production rate in cage
#     x = np.max(-1.0*np.asarray([data['q6'],data['q7'],data['q8'],data['q9']
#                                 ]),axis=0)/1.66667e-8
#     prob2drcat(x=x,
#                cat=cat_ther,
#                nbin=200,
#                names = ['R', 'O', 'E', 'N'],
#                labels = ['R = in %.2f yr' %(ts[3]), 'O = in %.2f yr' %(ts[ti]), 'E = in %.2f yr' %(ts[-1]), 'N = no breakthrough'],
#                x_label= 'Peak Cage Production Rate (mL/min)',
#                y_label= 'Ratios by Category')
#     plt.tight_layout()
#     plt.savefig('probbreakpro.png', format='png')

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

