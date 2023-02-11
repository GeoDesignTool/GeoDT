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
mLmin = 1.66667e-8 #m3/s
gal = 1.0/264.172 #m3

# *********************************************************************
#### each vs each vs Guggliemi
# *********************************************************************
def hist2d(x=[],y=[],bins=[],log=False,heat=True,x_label='',y_label=''):
    # *********************************************************************
    # bins
    if not(bins):
        bins = [50,25]
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
    # create figure
    fig = pylab.figure(figsize=(10.3,5),dpi=100)
    ax1 = fig.add_subplot(111)
    # return values to normal space
    if log:
        xedges = 10.0**xedges
        yedges = 10.0**yedges
        x_val = 10.0**x_val
        y_val = 10.0**y_val
    # heatmap
    if heat:
        if log:
            for x_i in range(1,len(xedges)):
                sc = ax1.scatter(x = np.ones(len(yedges)-1)*xedges[x_i],
                                  y = yedges[1:],
                                  c = H[x_i-1,:],
                                  cmap='gist_stern_r')
            plt.colorbar(mappable=sc,label='Frequency (samples)',orientation='vertical')
        else:
            ax1.imshow(np.transpose(H[:,:]),aspect='auto',cmap='hot',extent=(xedges[0],xedges[-1],yedges[0],yedges[-1]),origin='lower',zorder=-1)
        #ax1.imshow(np.transpose(H[:,:]),aspect='auto',cmap='hot',extent=(xedges[0],xedges[-1],yedges[0],yedges[-1]),origin='lower',zorder=-1)
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
    ax1.plot(xedges[1:],vis[1,:],'-',color='grey')
    # plot confidence interval
    ax1.fill_between(xedges[1:],vis[0,:],vis[2,:],color='grey', alpha=0.3)
    # plot data
    if not(heat):
        ax1.plot(x_val,y_val,'.',label='results',color='grey')
    # ax1.plot(x_val,y_val,'.',label='results',color='grey')
    
    # formatting
    #ax1.legend(loc='upper left', fancybox=True, prop={'size':8}, shadow=False, ncol=1, numpoints=1)
    ax1.set_xlabel(x_label,fontsize=10)
    ax1.set_ylabel(y_label,fontsize=10)
    if log:
        ax1.set_yscale('log')
        ax1.set_xscale('log')
    
    plt.tight_layout()

# *********************************************************************
#### 2D histogram with probability ratios
# *********************************************************************
def prob2drcat(x=[],cat=[],nbin=50,names=[],labels=[],x_label='',y_label='',log=False):
    #logscale
    if log:
        x = np.log10(x)
    
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
        prob[b,:] = prob[b,:]/np.sum(prob[b,:])
        
    #return to normal units
    if log:
        x = 10.0**x
        bins = 10.0**bins
    
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
    if log:
        ax1.set_xscale('log')
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
filename = 'inputs_results_FORGE.txt'
data = np.recfromcsv(filename,delimiter=',',filling_values=np.nan,deletechars='()',case_sensitive=True,names=True)

print('samples = %i' %(len(data)))

#prefilter
if False:
    data = data[:5000]
if False:
    keep = np.zeros(len(data),dtype=bool)
    criteria = [['Qinj',0.0005,0.040],
                ['w_spacing',80.0,150.0]
                #['Vinj',80.0e-3,2000.0e-3]
                #['Qinj',3000*mLmin,100000*mLmin]
                ]
    for i in range(0,len(criteria)):
        data = data[data[criteria[i][0]] > criteria[i][1]]
        data = data[data[criteria[i][0]] < criteria[i][2]]
if False:
    keep = np.zeros(len(data),dtype=bool)
    criteria = [['w_intervals',2.9,11.1]
                #['Vinj',80.0e-3,2000.0e-3]
                #['Qinj',3000*mLmin,100000*mLmin]
                ]
    for i in range(0,len(criteria)):
        data = data[data[criteria[i][0]] > criteria[i][1]]
        data = data[data[criteria[i][0]] < criteria[i][2]]
if False: #filter out non-caged models
    data['q0'][data['q0'] < 1.0e-7] = 1.0e-7
    recovery2 = (-(data['q1'])/data['q0'])
    data = data[(recovery2 > 0.8)*(recovery2 < 1.25)]
if False: #filter out failed models
    data['q0'][data['q0'] < 1.0e-7] = 1.0e-7
    data = data[data['q0']<(1.1*data['Qinj'])]
    data = data[data['q0']>(0.91*data['Qinj'])]

#seismic magnitude softener
#data['max_quake'] = np.min([data['max_quake'], (np.log10(3.0e10*data['Vinj'])-9.1)/1.5],axis=0)
# data['max_quake'] = data['max_quake'] -2.5
    
        
print('filtered samples = %i' %(len(data)))

#restructure timeseries data
names = data.dtype.names
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
T_lo = np.min([np.min(data['Tinj']),np.min(data['BH_T'])])
T_to_h, h_to_T = enthalpy(np.max(data['Tinj'])+273.15,np.max(data['BH_T']),np.average(data['BH_P'])*10**-6)
Tpro = np.zeros((len(data),TimeSteps),dtype=float)
for t in range(0,TimeSteps):
    Tpro[:,t] = h_to_T[0]*hpro[:,t]**3.0 + h_to_T[1]*hpro[:,t]**2.0 + h_to_T[2]*hpro[:,t] + h_to_T[3]

#placeholder for electrical energy generation with binary cycle
Eout = np.asarray(Pout)
np.nan_to_num(Eout,copy=False,nan=0.0)

#identify result categories
#producing (P = producting, N = non-producing, M = marginal)
cat_pro = np.full(len(data),'M')
cat_pro[Eout[:,-1] > 1.0e3] = 'P'
cat_pro[Eout[:,-1] < 0.1e3] = 'N'

#all well containment (C = caged, L = leaking, G = gaining, O = other)
cat_cag = np.full(len(data),'C')
np.nan_to_num(data['qinj'],copy=False,nan=1e-7)
np.nan_to_num(data['qpro'],copy=False,nan=1e-8)
data['qinj'][data['qinj'] < 1e-7] = 1e-7
cat_cag[(-data['qpro']/data['qinj']) < 0.9] = 'O'
cat_cag[(-data['qpro']/data['qinj']) > 1.11] = 'O'
cat_cag[(-data['qpro']/data['qinj']) < 0.5] = 'L'
cat_cag[(-data['qpro']/data['qinj']) > 2.0] = 'G'

#seismic vs not
cat_seis = np.full(len(data),'I')
cat_seis[(data['max_quake'] > 1.0)] = 'R'
cat_seis[(data['max_quake'] > 3.0)] = 'A'
cat_seis[(data['max_quake'] > 4.0)] = 'C'

#shear vs tensile
cat_shea = np.full(len(data),'U')
cat_shea[(data['hfstim'] > 0.0)] = 'H'
cat_shea[(data['nfstim'] > 0.0)] = 'S'
cat_shea[(data['nfstim']*data['hfstim'] > 0.0)] = 'M'

#thermal breakthrough by 10% reduction in fluid temperature
tcrit = 0.1*(data['BH_T']-(data['Tinj']+273.15))
ti = int(0.2*TimeSteps)+1
cat_ther = np.full(len(data),'N')
cat_ther[np.abs((data['BH_T']-Tpro[:,-1])) > tcrit] = 'E' #breakthrough before final timestep
cat_ther[np.abs((data['BH_T']-Tpro[:,ti])) > tcrit] = 'O' #moderate breakthrough
cat_ther[np.abs((data['BH_T']-Tpro[:,3])) > tcrit] = 'R' #breakthrough in first time step

#fracture size
cat_fsize = np.full(len(data),'T')
cat_fsize[data['dia_last'] > 0.9*data['w_spacing']] = 'O' #fracture exceeds design limit
cat_fsize[data['dia_last'] > 1.5*data['w_spacing']] = 'L' #fracture exceeds design limit
cat_fsize[data['dia_last'] > 3.0*data['w_spacing']] = 'E' #fracture exceeds design limit

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

# #power output histograms
# x_lab = 'BH_T'
# y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
# x_bin = np.linspace(np.min(data[x_lab]),np.max(data[x_lab]),int(0.5*nbin)+1)
# hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='reservoir temp (K)',y_label='electrical power output (kW)')

# x_lab = 'Qinj'
# y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
# x_bin = np.linspace(0.0001,0.1,int(0.5*nbin)+1)
# hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='interval injection rate (m3/s)',y_label='electrical power output (kW)')

# x_lab = 'w_spacing'
# y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
# x_bin = np.linspace(np.min(data[x_lab]),np.max(data[x_lab]),int(0.5*nbin)+1)
# hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='well spacing (m)',y_label='electrical power output (kW)')

# x_lab = 'w_count'
# y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
# x_bin = np.linspace(0.5,np.max(data[x_lab])+0.5,int(np.max(data[x_lab])+1.2))
# hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='number of production wells',y_label='electrical power output (kW)')

# x_lab = 'w_intervals'
# y_bin = np.linspace(0.01e4,100.0e3,nbin+1)
# x_bin = np.linspace(0.5,np.max(data[x_lab])+0.5,int(np.max(data[x_lab])+1.2))
# hist2d(x=data[x_lab],y=Pout[:,-1],bins=[x_bin,y_bin],log=False,heat=True,x_label='number of injection intervals',y_label='electrical power output (kW)')

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
lohi = [-0.1,0.1]        
# for n in range(0,2):
#     na = 'q%i' %(n)
#     data_n[na] = (data[na]-lohi[0])/(lohi[1]-lohi[0])
#     data_n[na][data_n[na] > 1.0] = 1.0
#     data_n[na][data_n[na] < 0.0] = 0.0
# na = 'qinj'
# data_n[na] = (data[na]-lohi[0])/(lohi[1]-lohi[0])
# data_n[na][data_n[na] > 1.0] = 1.0
# data_n[na][data_n[na] < 0.0] = 0.0
# na = 'qpro'
# data_n[na] = (data[na]-lohi[0])/(lohi[1]-lohi[0])
# data_n[na][data_n[na] > 1.0] = 1.0
# data_n[na][data_n[na] < 0.0] = 0.0

# ****************************************************************************
#### hydroshear boxplot normalized
# ****************************************************************************
if True:
    #data columns of interest
    column = ['Qinj','w_spacing','w_proportion','w_phase','w_toe','w_skew','w_intervals','dPp','BH_P','pinj','qinj','qpro']
    
    # ****************************************************************************
    #filter to power producing
    keep = np.zeros(len(data),dtype=bool)
    for i in range(0,len(data)):
        if (cat_pro[i] == 'P'):
            keep[i] = True
    data_PC = data_n[keep]
    #boxplot
    fig = pylab.figure(figsize=(16,10),dpi=100)
    ax1 = fig.add_subplot(411)
    df = pd.DataFrame(data_PC,columns=names)
    boxplot = df.boxplot(grid=False,column=column)
    ax1.set_ylabel('Power >1 MWe (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
    # ****************************************************************************
    #filter to contained
    keep = np.zeros(len(data),dtype=bool)
    for i in range(0,len(data)):
        if (cat_cag[i] == 'C'):
            keep[i] = True
    data_PL = data_n[keep]
    
    #boxplot
    ax2 = fig.add_subplot(412)
    df = pd.DataFrame(data_PL,columns=names)
    boxplot = df.boxplot(grid=False,column=column)
    ax2.set_ylabel('90%% < Recov < 111%% (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
    # ****************************************************************************
    #filter to rapid breakthrough
    keep = np.zeros(len(data),dtype=bool)
    for i in range(0,len(data)):
        if cat_ther[i] == 'R':
            keep[i] = True
    data_S = data_n[keep]
    
    #boxplot
    ax3 = fig.add_subplot(413)
    df = pd.DataFrame(data_S,columns=names)
    boxplot = df.boxplot(grid=False,column=column)
    ax3.set_ylabel('Breakthru <%.2f yr (%.1f%%)' %(ts[3],100.0*np.sum(keep)/len(keep)))
    
    # ****************************************************************************
    #filter to non-seismogenic
    keep = np.zeros(len(data),dtype=bool)
    for i in range(0,len(data)):
        if (cat_seis[i] == 'I'):
            keep[i] = True
    data_NL = data_n[keep]
    
    #boxplot
    ax4 = fig.add_subplot(414)
    df = pd.DataFrame(data_NL,columns=names)
    boxplot = df.boxplot(grid=False,column=column)
    ax4.set_ylabel('Quake <1.0 Mw (%.1f%%)' %(100.0*np.sum(keep)/len(keep)))
    
    # ****************************************************************************
    #### formatting
    # ****************************************************************************
    #print value ranges
    title = ''
    print('\n*** BOXPLOT VALUE RANGES ***')
    for n in column:
        print([n,np.min(data[n]),np.average(data[n]),np.max(data[n])])
        title = title + '%s [%.2e,%.2e]; ' %(n, np.min(data[n]), np.max(data[n]))
    ax1.set_title(title, fontsize=6)
    plt.tight_layout()
    
    
    
# ****************************************************************************
#### pie charts
# ****************************************************************************
if True:
    # ****************************************************************************
    #### pie chart: contained
    # ****************************************************************************
    labels = ['C', 'L', 'O', 'G']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_cag[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['90-111%', '<50%','Other', '>200%']
    explode = (0,0,0,0)
    fig = pylab.figure(figsize=(10,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    ax1.set_title('Flow Contained (%i samples)' %(np.sum(1*(data['type_last']==2))))
    plt.tight_layout()
    plt.savefig('contain.png', format='png')
    pylab.close()

    # ****************************************************************************
    #### pie chart: breakthrough
    # ****************************************************************************
    labels = ['R', 'O', 'E', 'N']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_ther[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['↓%.2f C in %.2f yr' %(tcrit[-1],ts[3]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[ti]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1]), '<↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1])]
    explode = (0,0,0,0)
    fig = pylab.figure(figsize=(10,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    ax1.set_title('Thermal Breakthrough (%i samples)' %(np.sum(1*(data['type_last']==2))))
    plt.tight_layout()
    plt.savefig('break.png', format='png')
    pylab.close()

    # ****************************************************************************
    #### pie chart: seismic
    # ****************************************************************************
    labels = ['I', 'R', 'A', 'C']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_seis[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['Mw < 1.0','Mw > 1.0','Mw > 3.0','Mw > 4.0']
    explode = (0,0,0,0)
    fig = pylab.figure(figsize=(10,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    ax1.set_title('Seismicity Maximum (%i samples)' %(np.sum(1*(data['type_last']==2))))
    plt.tight_layout()
    plt.savefig('seis.png', format='png')
    pylab.close()

    # ****************************************************************************
    #### pie chart: shear
    # ****************************************************************************
    labels = ['S', 'M', 'H', 'U']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_shea[j] == labels[i]:
                keep[j] = 1
        sizes += [np.sum(keep)/len(keep)]
    labels = ['Hydroshear','Mixed', 'Hydrofrac', 'None']
    explode = (0,0,0,0)
    fig = pylab.figure(figsize=(10,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    ax1.set_title('Hydroshear vs Hydrofrac (%i samples)' %(np.sum(1*(data['type_last']==2))))
    plt.tight_layout()
    plt.savefig('shear.png', format='png')
    pylab.close()

    # ****************************************************************************
    #### pie chart: fracture size
    # ****************************************************************************
    labels = ['O', 'T', 'L', 'E']
    sizes = []
    for i in range(0,len(labels)):
        keep = np.zeros(len(data),dtype=int)
        for j in range(0,len(data)):
            if cat_fsize[j] == labels[i]:
                keep[j] = 1
            if data['type_last'][j] != 2:
                keep[j] = 0
        sizes += [np.sum(keep)/len(keep)]
    labels = ['>0.9x w_s','<0.9x w_s', '>1.5x w_s', '>3x w_s']
    explode = (0,0,0,0)
    fig = pylab.figure(figsize=(10,5),dpi=100)
    ax1 = fig.add_subplot(111)
    ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
    ax1.axis('equal')
    ax1.set_title('Freshest Fracture Radius (%i samples)' %(np.sum(1*(data['type_last']==2))))
    plt.tight_layout()
    plt.savefig('hfsize.png', format='png')
    pylab.close()
    
    # ****************************************************************************
    #### pie chart: power objective function
    # ****************************************************************************
    #Generation 1
    if False:
        #power objective function
        Tref = 120+273.15
        efficiency = 0.13
        href = T_to_h[0]*Tref**3.0 + T_to_h[1]*Tref**2.0 + T_to_h[2]*Tref + T_to_h[3]
        Tbas = 95+273.15
        hbas = T_to_h[0]*Tbas**3.0 + T_to_h[1]*Tbas**2.0 + T_to_h[2]*Tbas + T_to_h[3]
        size = np.zeros(len(dhout))
        hour = (ts[1]-ts[0])*8760 #hours per timestep
        for i in range(0,len(dhout[0])):
            size = size + hour*efficiency*(dhout[:,i]+(hbas-href)*data['mpro'])
        size = 10.0**-6.0*size/LifeSpan
        cat_obj = np.full(len(data),'M')
        cat_obj[size[:] > 8.0] = 'P' #8.7 GWe-h/yr = 1 MWe
        cat_obj[size[:] > 80.0] = 'H'
        cat_obj[size[:] < 0.8] = 'N'
        holdthis = np.asarray(size)
        
        #Cumulative Objective Function - Power - Datathon
        labels = ['H', 'P', 'M', 'N']
        sizes = []
        for i in range(0,len(labels)):
            keep = np.zeros(len(data),dtype=int)
            for j in range(0,len(data)):
                if cat_obj[j] == labels[i]:
                    keep[j] = 1
                # if data['hfstim'][j] > 0:
                #     keep[j] = 0
            sizes += [np.sum(keep)/len(keep)]
        # sizes[-1] = 1.0 - np.sum(sizes[:-1])
        labels = ['>80','>8.0', '>0.8', '<0.8']
        explode = (0,0,0,0)
        fig = pylab.figure(figsize=(10,5),dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        ax1.axis('equal')
        ax1.set_title('Cumularive Annual Objective Power Production (%.2f yr)' %(ts[-1]))
        plt.tight_layout()
        plt.savefig('objective.png', format='png')
        pylab.close()
    
    #Generation 2
    if True:
        # pie chart: NPV
        sizes = []
        labels = ['>$50M', '%$-50M to %$0M','%$0M to %$50M', '<-$50M']
        keep = np.zeros(4,dtype=int)
        for j in range(0,len(data)):
            if data['NPV'][j] < -50.0e6:
                keep[3] += 1
            elif data['NPV'][j] < 0.0:
                keep[1] += 1
            elif data['NPV'][j] < 50.0e6:
                keep[2] += 1
            else:
                keep[0] += 1
            sizes = 1.0*keep/np.sum(keep)
        explode = (0,0,0,0)
        fig = pylab.figure(figsize=(10,5),dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        ax1.axis('equal')
        ax1.set_title('Net Present Value (%i samples)' %(np.sum(keep)))
        plt.tight_layout()
        plt.savefig('npv.png', format='png')
        pylab.close() 
        
        # pie chart: injection pressure
        sizes = []
        labels = ['<s3', '>s3']
        keep = np.zeros(2,dtype=int)
        for j in range(0,len(data)):
            if data['pinj'][j]*10**6 < data['s3'][j]:
                keep[0] += 1
            else:
                keep[1] += 1
            sizes = 1.0*keep/np.sum(keep)
        explode = (0,0)
        fig = pylab.figure(figsize=(10,5),dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        ax1.axis('equal')
        ax1.set_title('Injection Pressure (%i samples)' %(np.sum(keep)))
        plt.tight_layout()
        plt.savefig('pinj.png', format='png')
        pylab.close()
        
        # pie chart average net positive power (flash, binary, pump)
        size = np.zeros(len(Pout))
        vals = 0.0+Pout
        vals[vals<0.0] = 0.0 #assume plant is offline when power generation is zero
        dt = (ts[1]-ts[0]) #hours per timestep
        for i in range(0,len(vals[0])):
            size = size + dt*vals[:,i]
        size = size/LifeSpan        
        sizes = []
        labels = ['>30 MWe', '1.0 to 10 MWe','10 to 30 MWe', '<1.0 MWe']
        keep = np.zeros(4,dtype=int)
        for j in range(0,len(data)):
            if size[j] < 1.0e3:
                keep[3] += 1
            elif size[j] < 10.0e3:
                keep[1] += 1
            elif size[j] < 30.0e3:
                keep[2] += 1
            else:
                keep[0] += 1
            sizes = 1.0*keep/np.sum(keep)
        explode = (0,0,0,0)
        fig = pylab.figure(figsize=(10,5),dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        ax1.axis('equal')
        ax1.set_title('Average Net Power (%i samples)' %(np.sum(keep)))
        plt.tight_layout()
        plt.savefig('NPout.png', format='png')
        pylab.close()         
        
    
# ****************************************************************************
#### category probability plots
# ****************************************************************************
if True:
    # probability plot of breakthrough vs production pressure
    x = (data['BH_P']+data['dPp'])/MPa #m3/s
    prob2drcat(x=x,
               cat=cat_ther,
               nbin=30,
               names = ['R', 'O', 'E', 'N'],
               labels = ['↓%.2f C in %.2f yr' %(tcrit[-1],ts[3]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[ti]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1]), '<↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1])],
               x_label= 'Production Pressure (MPa)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probbreakback.png', format='png')
    pylab.close()
    
    # probability plot of breakthrough vs injection rate
    x = data['Qinj']/1.66667e-8 #m3/s
    prob2drcat(x=x, log=True,
               cat=cat_ther,
               nbin=30,
               names = ['R', 'O', 'E', 'N'],
               labels = ['↓%.2f C in %.2f yr' %(tcrit[-1],ts[3]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[ti]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1]), '<↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1])],
               x_label= 'Injection Rate (mL/min)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probbreak.png', format='png')
    pylab.close()

    # probability plot of cage vs injection rate
    x = data['Qinj']/1.66667e-8 #m3/s
    prob2drcat(x=x, log=True,
               cat=cat_cag,
               nbin=30,
               names = ['C','O','L','G'],
               labels = ['Caged 80-125%','Other', 'Leaking <40%', 'Gaining >250%'],
               x_label= 'Injection Rate (mL/min)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probcage4.png', format='png')
    pylab.close()

    # probability plot of fracture radius vs injection rate
    x = data['Qinj']/1.66667e-8 #m3/s
    prob2drcat(x=x, log=True,
               cat=cat_fsize,
               nbin=30,
               names = ['O', 'T', 'L', 'E'],
               labels = ['>0.9x w_s','<0.9x w_s', '>1.5x w_s', '>3x w_s'],
               x_label= 'Injection Rate (mL/min)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probfsize.png', format='png')
    pylab.close()
    
    # probability plot of fracture radius vs injection volume
    x = data['Vinj']*1.0e3 #L
    prob2drcat(x=x, log=True,
               cat=cat_fsize,
               nbin=17,
               names = ['O', 'T', 'L', 'E'],
               labels = ['>0.9x w_s','<0.9x w_s', '>1.5x w_s', '>3x w_s'],
               x_label= 'Injection Volume (L)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probfsize.png', format='png')
    pylab.close()
    
    # probability plot of power vs injection rate
    x = data['Qinj']/1.66667e-8 #m3/s
    prob2drcat(x=x, log=True,
               cat=cat_pro,
               nbin=30,
               names = ['P', 'M', 'N'],
               labels = ['>1 MWe','>0.1 MWe', '<0.1 MWe'],
               x_label= 'Injection Rate (mL/min)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probpowerrate.png', format='png')
    pylab.close()
    
    # probability plot of power vs well spacing
    x = data['w_spacing'] #m
    prob2drcat(x=x, log=False,
               cat=cat_pro,
               nbin=30,
               names = ['P', 'M', 'N'],
               labels = ['>1 MWe','>0.1 MWe', '<0.1 MWe'],
               x_label= 'Well Spacing (m)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probpowerspacing.png', format='png')
    pylab.close()
    
    
    # probability plot of quake vs injection rate
    x = data['Qinj']/1.66667e-8 #m3/s
    prob2drcat(x=x, log=True,
               cat=cat_seis,
               nbin=30,
               names = ['I', 'R', 'A', 'C'],
               labels = ['<1 Mw','>1 Mw','>3 Mw','>4 Mw'],
               x_label= 'Injection Rate (mL/min)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probseisrate.png', format='png')
    pylab.close()
    
    # probability plot of quake vs well spacing
    x = data['w_spacing'] #m
    prob2drcat(x=x, log=False,
               cat=cat_seis,
               nbin=30,
               names = ['I', 'R', 'A', 'C'],
               labels = ['<1 Mw','>1 Mw','>3 Mw','>4 Mw'],
               x_label= 'Well Spacing (m)',
               y_label= 'Ratios by Category')
    plt.tight_layout()
    plt.savefig('probseisspacing.png', format='png')
    pylab.close()

    # # probability plot of breakthrough vs peak production rate
    # x = -1.0*data['q1']/1.66667e-8
    # prob2drcat(x=x, log=True,
    #            cat=cat_ther,
    #            nbin=50,
    #            names = ['R', 'O', 'E', 'N'],
    #            labels = ['↓%.2f C in %.2f yr' %(tcrit[-1],ts[3]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[ti]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1]), '<↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1])],
    #            x_label= 'Peak Overall Production Rate (mL/min)',
    #            y_label= 'Ratios by Category')
    # plt.tight_layout()
    # plt.savefig('probbreakpro.png', format='png')
    
    # # probability plot of breakthrough vs peak production rate in cage
    # x = -1.0*data['q1']/1.66667e-8
    # x[x < 1.0e-7] = 1.0e-7/1.66667e-8
    # prob2drcat(x=x, log=True,
    #            cat=cat_ther,
    #            nbin=50,
    #            names = ['R', 'O', 'E', 'N'],
    #            labels = ['↓%.2f C in %.2f yr' %(tcrit[-1],ts[3]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[ti]), '↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1]), '<↓%.2f C in %.2f yr' %(tcrit[-1],ts[-1])],
    #            x_label= 'Peak Cage Production Rate (mL/min)',
    #            y_label= 'Ratios by Category')
    # plt.tight_layout()
    # plt.savefig('probbreakpro.png', format='png')



# ****************************************************************************
#### 2D probability plots (heat maps)
# ****************************************************************************
if False:
    # 2D probability plot of recovery rate vs injection rate
    recovery2 = []
    recovery2 = -data['recovery']
    recovery2[recovery2>2.5] = 2.5
    recovery2[recovery2<0.4] = 0.4
    hist2d(x=data['Qinj']/1.66667e-8,y=recovery2,bins=[50,25],log=True,heat=True,x_label='Injection Rate (mL/min)',y_label='Recovery Ratio (caged)')
    plt.tight_layout()
    plt.savefig('prob2drecovery2.png', format='png')

    # 2D probability plot of fracture size vs injection rate
    size = data['dia_last']/2.0
    hist2d(x=data['Qinj']/1.66667e-8,y=size,bins=[50,25],log=True,heat=True,x_label='Injection Rate (mL/min)',y_label='Fracture Radius (m)')
    plt.tight_layout()
    plt.savefig('prob2dfsize_rate.png', format='png')

    # 2D probability plot of fracture size vs injection volume
    size = data['dia_last']/2.0
    hist2d(x=data['Vinj']*1.0e3,y=size,bins=[50,25],log=True,heat=True,x_label='Injection Volume (L)',y_label='Fracture Radius (m)')
    plt.tight_layout()
    plt.savefig('prob2dfsize_vol.png', format='png')
    
    # 2D probability plot of crude usable enthalpy vs injection rate
    # #morphed negative log plot
    # Tref = 120+273.15
    # efficiency = 0.13
    # href = T_to_h[0]*Tref**3.0 + T_to_h[1]*Tref**2.0 + T_to_h[2]*Tref + T_to_h[3]
    # Tbas = 95+273.15
    # hbas = T_to_h[0]*Tbas**3.0 + T_to_h[1]*Tbas**2.0 + T_to_h[2]*Tbas + T_to_h[3]
    # size = efficiency*(dhout[:,-1]+(hbas-href)*data['mpro'])
    # size = np.sign(size)*np.log10(np.abs(size)+0.1)
    # dims = data['Qinj']
    # dims = np.sign(dims)*np.log10(np.abs(dims)+0.001)
    # hist2d(x=dims,y=size,bins=[50,25],log=False,heat=True,x_label='Log10 Injection Rate (m3/s)',y_label='Log10 Usable Power Out (kW)')
    # plt.tight_layout()
    # plt.savefig('prob2dpower1.png', format='png')
    #truncated plot
    Tref = 120+273.15
    efficiency = 0.13
    href = T_to_h[0]*Tref**3.0 + T_to_h[1]*Tref**2.0 + T_to_h[2]*Tref + T_to_h[3]
    Tbas = 95+273.15
    hbas = T_to_h[0]*Tbas**3.0 + T_to_h[1]*Tbas**2.0 + T_to_h[2]*Tbas + T_to_h[3]
    size = efficiency*(dhout[:,-1]+(hbas-href)*data['mpro'])
    size[size < 10.0] = 10.0 
    dims = data['Qinj']
    size[size < 0.0005] = 0.0005 
    hist2d(x=dims,y=size,bins=[50,25],log=True,heat=True,x_label='Injection Rate (m3/s)',y_label='Usable Power Out (kW)')
    plt.tight_layout()
    plt.savefig('prob2dpower2.png', format='png')

    # 2D probability plot of positive power on well spacing versus flow rate plot
    dim2 = data['w_spacing']
    hist2d(x=dim2,y=size,bins=[50,25],log=True,heat=True,x_label='Well Spacing (m)',y_label='Usable Power Out (kW)')
    plt.tight_layout()
    plt.savefig('prob2dpower3.png', format='png')  
    
    # # 2D probability plot of positive power on well spacing versus flow rate plot
    # dim2 = dims[size > 10.1]
    # siz2 = data['w_spacing'][size > 10.1]
    # hist2d(x=dim2,y=siz2,bins=[50,25],log=True,heat=True,x_label='Injection Rate (m3/s)',y_label='Well Spacing (m)')
    # plt.tight_layout()
    # plt.savefig('prob2dpower3.png', format='png')    

if True:
    #****************************************************************************
    ### 2D point color plots
    #****************************************************************************
    #Breakthrough time as function of spacing and flow rate
    tb = np.zeros(len(data))
    for i in range(0,len(tb)):
        hold = np.argwhere(np.abs((data['BH_T'][i]-Tpro[i,:])) > tcrit[i])
        if len(hold) < 1:
            tb[i] = data['TimeSteps'][i] + 1
        else:
            tb[i] = hold[0][0]
    tb = tb*(ts[1]-ts[0])
    fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
    ax1 = fig.add_subplot(111)
    sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=tb,cmap=plt.cm.get_cmap('YlOrRd'))
    plt.colorbar(mappable=sc,label='10% Breakthrough Time (yr)',orientation='vertical')
    ax1.set_xlabel('Well Spacing (m)',fontsize=10)
    ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
    ax1.set_yscale('log')
    plt.tight_layout()
    plt.savefig('BT_v_S_v_Q.png', format='png')

    #Seismic Risk
    fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
    ax1 = fig.add_subplot(111)
    sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=data['max_quake'],cmap=plt.cm.get_cmap('rainbow'))
    plt.colorbar(mappable=sc,label='Max %"Possible%" Quake (Mw)',orientation='vertical')
    ax1.set_xlabel('Well Spacing (m)',fontsize=10)
    ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
    ax1.set_yscale('log')
    plt.tight_layout()
    plt.savefig('Mw_v_S_v_Q.png', format='png')
    
    #Generation 1
    if False:
        #objective power output as function of spacing and flow rate
        Tref = 120+273.15
        efficiency = 0.13
        href = T_to_h[0]*Tref**3.0 + T_to_h[1]*Tref**2.0 + T_to_h[2]*Tref + T_to_h[3]
        Tbas = 95+273.15
        hbas = T_to_h[0]*Tbas**3.0 + T_to_h[1]*Tbas**2.0 + T_to_h[2]*Tbas + T_to_h[3]
        size = efficiency*(dhout[:,4]+(hbas-href)*data['mpro'])
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=size[:],cmap=plt.cm.get_cmap('bwr'))
        plt.colorbar(mappable=sc,label='%.1f Year Objective Power Output (kWe)' %(ts[4]),orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('OE_v_S_v_Q.png', format='png')
        
        #Rankine power output as function of spacing and flow rate
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=Eout[:,4],cmap=plt.cm.get_cmap('YlOrRd'))
        plt.colorbar(mappable=sc,label='%.1f Year Rankine Power Output (kWe)' %(ts[4]),orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('RE_v_S_v_Q.png', format='png')        
        
        #cumulative objective power output as function of spacing and flow rate
        size = np.zeros(len(dhout))
        hour = (ts[1]-ts[0])*8760 #hours per timestep
        for i in range(0,len(dhout[0])):
            size = size + hour*efficiency*(dhout[:,i]+(hbas-href)*data['mpro'])
        size = 10.0**-6.0*size/LifeSpan
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=size[:],cmap=plt.cm.get_cmap('coolwarm'))
        plt.colorbar(mappable=sc,label='%.1f Year Cumulative Objective Power Output (GWe-h/yr)' %(ts[-1]),orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('COE_v_S_v_Q.png', format='png')
    
        #revenue objective power output as function of spacing and flow rate
        size = np.zeros(len(dhout))
        hour = (ts[1]-ts[0])*8760 #hours per timestep
        for i in range(0,len(dhout[0])):
            size = size + hour*efficiency*(dhout[:,i]+(hbas-href)*data['mpro'])
        size = 0.18*10.0**-6.0*size/LifeSpan
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=size[:],cmap=plt.cm.get_cmap('coolwarm'))
        plt.colorbar(mappable=sc,label='Gross Sales Revenue at 0.18C/kWh (MUSD/yr)',orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('DOE_v_S_v_Q.png', format='png')

    #Generation 2
    if True:
        #NPV vs flow rate and spacing
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=data['NPV'],vmin=-50e6,vmax=50e6,cmap=plt.cm.get_cmap('rainbow_r'))
        plt.colorbar(mappable=sc,label='Net Present Value ($)',orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('NPV_v_S_v_Q.png', format='png')
        
        #Cumulative Pout per Year (binary + flash - pumping) vs flow rate and spacing
        size = np.zeros(len(Pout))
        vals = 0.0+Pout
        #vals[vals<0.0] = 0.0 #assume plant is offline when power generation is zero
        dt = (ts[1]-ts[0]) #hours per timestep
        for i in range(0,len(vals[0])):
            size = size + dt*vals[:,i]
        size = size/LifeSpan        
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=size,vmin=-10e3,vmax=10e3,cmap=plt.cm.get_cmap('rainbow_r'))
        plt.colorbar(mappable=sc,label='Average Net Power Output per Year (kW)',orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('APY_v_S_v_Q.png', format='png')        
        
        
    
if True:
    #****************************************************************************
    ### enthalpy timeseries
    #****************************************************************************
    fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
    ax1 = fig.add_subplot(111)
    for s in range(0,len(data)):
        ax1.plot(ts,hpro[s,:])
    ax1.set_xlabel('Time (yr)',fontsize=10)
    ax1.set_ylabel('Produced Enthalpy (kJ/kg)',fontsize=10)
    plt.tight_layout()
    plt.savefig('enthalpy_timeseries.png', format='png')
    
    #****************************************************************************
    ### power timeseries
    #****************************************************************************
    fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
    ax1 = fig.add_subplot(111)
    for s in range(0,len(data)):
        ax1.plot(ts,Pout[s,:])
    ax1.set_xlabel('Time (yr)',fontsize=10)
    ax1.set_ylabel('Produced Power (kW)',fontsize=10)
    plt.tight_layout()
    plt.savefig('power_timeseries.png', format='png')

if False:
    #****************************************************************************
    ### power timeseries
    #****************************************************************************
    #objective power output as function of spacing and flow rate
    Tref = 120+273.15
    efficiency = 0.13
    href = T_to_h[0]*Tref**3.0 + T_to_h[1]*Tref**2.0 + T_to_h[2]*Tref + T_to_h[3]
    Tbas = 95+273.15
    hbas = T_to_h[0]*Tbas**3.0 + T_to_h[1]*Tbas**2.0 + T_to_h[2]*Tbas + T_to_h[3]
    fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
    ax1 = fig.add_subplot(111)
    for s in range(0,len(data)):
        Eout[s,:] = efficiency*(dhout[s,:]+(hbas-href)*data['mpro'][s])
        ax1.plot(ts,Eout[s,:])
    ax1.set_xlabel('Time (yr)',fontsize=10)
    ax1.set_ylabel('Objective Power Output (kWe)',fontsize=10)
    plt.tight_layout()
    plt.savefig('objective_timeseries.png', format='png')
    
if True:
    #****************************************************************************
    ### filter to optimized flow rate & spacing
    #****************************************************************************
    #prefilter
    if False:
        data = data[:5000]
    if True:
        print('Filtering data to optimal flow rate range')
        keep = np.zeros(len(data),dtype=bool)
        # data = data[(data['Qinj']>(1e-5*data['w_spacing']))]
        # data = data[(data['Qinj']<(data['w_count']*2e-10*data['w_spacing']**3.0))]
        data = data[(data['Qinj']>(0.005))]
        data = data[(data['Qinj']<(0.033))]
        data = data[(data['Qinj']<(data['w_count']*5e-4*10.0**(3e-3*data['w_spacing'])))]
        #data = data[(data['Qinj']<(data['w_count']*1.5e-10*data['w_spacing']**3.0))]
        # data = data[(data['w_spacing']<(650.0))]
    
    print('filtered samples = %i' %(len(data)))
    
    #restructure timeseries data
    names = data.dtype.names
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
            
    #Generation 2
    if True:
        #NPV vs flow rate and spacing
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=data['NPV'],vmin=-50e6,vmax=50e6,cmap=plt.cm.get_cmap('rainbow_r'))
        plt.colorbar(mappable=sc,label='Net Present Value ($)',orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('NPV_v_S_v_Q_optQinj.png', format='png')
        
        #Cumulative Pout per Year (binary + flash - pumping) vs flow rate and spacing
        size = np.zeros(len(Pout))
        vals = 0.0+Pout
        #vals[vals<0.0] = 0.0 #assume plant is offline when power generation is zero
        dt = (ts[1]-ts[0]) #hours per timestep
        for i in range(0,len(vals[0])):
            size = size + dt*vals[:,i]
        size = size/LifeSpan        
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=size,vmin=-10e3,vmax=10e3,cmap=plt.cm.get_cmap('rainbow_r'))
        plt.colorbar(mappable=sc,label='Average Net Power Output per Year (kW)',orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('APY_v_S_v_Q_optQinj.png', format='png')

    #Generation 2
    if True:
        # pie chart: NPV
        sizes = []
        labels = ['>$50M', '%$-50M to %$0M','%$0M to %$50M', '<-$50M']
        keep = np.zeros(4,dtype=int)
        for j in range(0,len(data)):
            if data['NPV'][j] < -50.0e6:
                keep[3] += 1
            elif data['NPV'][j] < 0.0:
                keep[1] += 1
            elif data['NPV'][j] < 50.0e6:
                keep[2] += 1
            else:
                keep[0] += 1
            sizes = 1.0*keep/np.sum(keep)
        explode = (0,0,0,0)
        fig = pylab.figure(figsize=(10,5),dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        ax1.axis('equal')
        ax1.set_title('Net Present Value (%i samples)' %(np.sum(keep)))
        plt.tight_layout()
        plt.savefig('npv_optQinj.png', format='png')
        pylab.close() 
        
        # pie chart average net positive power (flash, binary, pump)
        size = np.zeros(len(Pout))
        vals = 0.0+Pout
        vals[vals<0.0] = 0.0 #assume plant is offline when power generation is zero
        dt = (ts[1]-ts[0]) #hours per timestep
        for i in range(0,len(vals[0])):
            size = size + dt*vals[:,i]
        size = size/LifeSpan        
        sizes = []
        labels = ['>30 MWe', '1.0 to 10 MWe','10 to 30 MWe', '<1.0 MWe']
        keep = np.zeros(4,dtype=int)
        for j in range(0,len(data)):
            if size[j] < 1.0e3:
                keep[3] += 1
            elif size[j] < 10.0e3:
                keep[1] += 1
            elif size[j] < 30.0e3:
                keep[2] += 1
            else:
                keep[0] += 1
            sizes = 1.0*keep/np.sum(keep)
        explode = (0,0,0,0)
        fig = pylab.figure(figsize=(10,5),dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        ax1.axis('equal')
        ax1.set_title('Average Net Power (%i samples)' %(np.sum(keep)))
        plt.tight_layout()
        plt.savefig('NPout_optQinj.png', format='png')
        pylab.close()
    
    #P10 statistics
    if True:
        #cumulative NPV function
        npvs = data['NPV']+0.0
        npvs = np.sort(npvs)
        nums = len(npvs)
        
        #collect percentiles
        out = []
        pcts = [0,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99,100]
        for p in pcts:
            out += [['npv%i' %(p), npvs[np.min([nums-1,int(1.0*nums*p/100)])]]]
        
        #save to file
        fname = "keys_" + filename
        out = list(map(list,zip(*out)))
        head = out[0][0]
        for i in range(1,len(out[0])):
            head = head + ',' + out[0][i]
        sdata = '%i' %(out[1][0])
        for i in range(1,len(out[1])):          
            if not out[1][i]:
                sdata = sdata + ',0.0' # ',nan'
            else:
                sdata = sdata + ',%.5e' %(out[1][i])
        try:
            with open(fname,'r') as f:
                test = f.readline()
            f.close()
            if test != '':
                with open(fname,'a') as f:
                    f.write(sdata + '\n')
                f.close()
            else:
                with open(fname,'a') as f:
                    f.write(head + '\n')
                    f.write(sdata + '\n')
                f.close()
        except:
            with open(fname,'a') as f:
                f.write(head + '\n')
                f.write(sdata + '\n')
            f.close()
        
        qcount = nums

if True:
    #****************************************************************************
    ### replace scenarios with pinj > s3 with losses
    #****************************************************************************    
    #recalculate NPV for cases with pinj > s3 (non-productive)
    for i in range(0,len(data)):
        if (data['pinj'][i]*10**6)>(data['s3'][i]):
            data['NPV'][i] = -data['cost_capital'][i]-data['risk_quakes'][i]
            
    #Generation 2
    if True:
        #NPV vs flow rate and spacing
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=data['NPV'],vmin=-50e6,vmax=50e6,cmap=plt.cm.get_cmap('rainbow_r'))
        plt.colorbar(mappable=sc,label='Net Present Value ($)',orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('NPV_v_S_v_Q_optQinj_pinj.png', format='png')
        
        # #Cumulative Pout per Year (binary + flash - pumping) vs flow rate and spacing
        # size = np.zeros(len(Pout))
        # vals = 0.0+Pout
        # #vals[vals<0.0] = 0.0 #assume plant is offline when power generation is zero
        # dt = (ts[1]-ts[0]) #hours per timestep
        # for i in range(0,len(vals[0])):
        #     size = size + dt*vals[:,i]
        # size = size/LifeSpan        
        # fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        # ax1 = fig.add_subplot(111)
        # sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=size,vmin=-10e3,vmax=10e3,cmap=plt.cm.get_cmap('rainbow_r'))
        # plt.colorbar(mappable=sc,label='Average Net Power Output per Year (kW)',orientation='vertical')
        # ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        # ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        # ax1.set_yscale('log')
        # plt.tight_layout()
        # plt.savefig('APY_v_S_v_Q_optQinj_pinj.png', format='png')

    #Generation 2
    if True:
        # pie chart: NPV
        sizes = []
        labels = ['>$50M', '%$-50M to %$0M','%$0M to %$50M', '<-$50M']
        keep = np.zeros(4,dtype=int)
        for j in range(0,len(data)):
            if data['NPV'][j] < -50.0e6:
                keep[3] += 1
            elif data['NPV'][j] < 0.0:
                keep[1] += 1
            elif data['NPV'][j] < 50.0e6:
                keep[2] += 1
            else:
                keep[0] += 1
            sizes = 1.0*keep/np.sum(keep)
        explode = (0,0,0,0)
        fig = pylab.figure(figsize=(10,5),dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        ax1.axis('equal')
        ax1.set_title('Net Present Value (%i samples)' %(np.sum(keep)))
        plt.tight_layout()
        plt.savefig('npv_optQinj_pinj.png', format='png')
        pylab.close() 
        
        # # pie chart average net positive power (flash, binary, pump)
        # size = np.zeros(len(Pout))
        # vals = 0.0+Pout
        # vals[vals<0.0] = 0.0 #assume plant is offline when power generation is zero
        # dt = (ts[1]-ts[0]) #hours per timestep
        # for i in range(0,len(vals[0])):
        #     size = size + dt*vals[:,i]
        # size = size/LifeSpan        
        # sizes = []
        # labels = ['>30 MWe', '1.0 to 10 MWe','10 to 30 MWe', '<1.0 MWe']
        # keep = np.zeros(4,dtype=int)
        # for j in range(0,len(data)):
        #     if size[j] < 1.0e3:
        #         keep[3] += 1
        #     elif size[j] < 10.0e3:
        #         keep[1] += 1
        #     elif size[j] < 30.0e3:
        #         keep[2] += 1
        #     else:
        #         keep[0] += 1
        #     sizes = 1.0*keep/np.sum(keep)
        # explode = (0,0,0,0)
        # fig = pylab.figure(figsize=(10,5),dpi=100)
        # ax1 = fig.add_subplot(111)
        # ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        # ax1.axis('equal')
        # ax1.set_title('Average Net Power (%i samples)' %(np.sum(keep)))
        # plt.tight_layout()
        # plt.savefig('NPout_optQinj_pinj.png', format='png')
        # pylab.close()
    
    #P10 statistics
    if True:
        #cumulative NPV function
        npvs = data['NPV']+0.0
        npvs = np.sort(npvs)
        nums = len(npvs)
        
        #collect percentiles
        out = []
        pcts = [0,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99,100]
        for p in pcts:
            out += [['npv%i' %(p), npvs[np.min([nums-1,int(1.0*nums*p/100)])]]]
        
        #save to file
        fname = "keys_pinj_" + filename
        out = list(map(list,zip(*out)))
        head = out[0][0]
        for i in range(1,len(out[0])):
            head = head + ',' + out[0][i]
        sdata = '%i' %(out[1][0])
        for i in range(1,len(out[1])):          
            if not out[1][i]:
                sdata = sdata + ',0.0' # ',nan'
            else:
                sdata = sdata + ',%.5e' %(out[1][i])
        try:
            with open(fname,'r') as f:
                test = f.readline()
            f.close()
            if test != '':
                with open(fname,'a') as f:
                    f.write(sdata + '\n')
                f.close()
            else:
                with open(fname,'a') as f:
                    f.write(head + '\n')
                    f.write(sdata + '\n')
                f.close()
        except:
            with open(fname,'a') as f:
                f.write(head + '\n')
                f.write(sdata + '\n')
            f.close()

if True:
    #****************************************************************************
    ### recompute power and NPV with different efficiency
    #****************************************************************************
    # ************************************************************************
    # energy generation - single flash rankine + simplified binary (Frash, 2020; Heberle and Bruggemann, 2010)
    # ************************************************************************
    def get_power(m_inj, #kg/s - injection mass flow
                  T_inj, #K - injection temperature
                  P_inj, #MPa - injection pressure
                  m_pro, #kg/s - production mass flow rate
                  h_pro, #kJ/kg - production enthalpy (can be an array)
                  P_whp, #MPa - steam flash pressure
                  P_exh, #MPa - turbine exhaust pressure (can assume 0.1 MPa)
                  effic=0.85, #General turbomachinery efficiency
                  detail=False, #print cycle infomation
                  plots=False): #plot results
        #initialization
        Flash_Power = []
        Binary_Power = []
        Pump_Power = []
        Net_Power = []
        
        #truncate pressures when superciritical
        pmax = 100.0
        P_inj = np.min([P_inj,pmax])
        P_whp = np.min([P_whp,pmax])
        P_exh = np.min([P_exh,pmax])
        
        #for each datapoint based on h_pro
        for t in range(0,len(h_pro)):
            # Surface Injection Well (5)
            T5 = T_inj # K
            P5 = P_inj # MPa
            state = therm(T=T5,P=P5)
            h5 = state.h # kJ/kg
            s5 = state.s # kJ/kg-K
            x5 = state.x # steam quality
            v5 = state.v # m3/kg
            
            # # Undisturbed Reservoir (r)
            # Tr = self.rock.BH_T #K
            # Pr = self.rock.BH_P/MPa # MPa
            # state = therm(T=Tr,P=Pr)
            # hr = state.h # kJ/kg
            # sr = state.s # kJ/kg-K
            # xr = state.x # steam quality
            # vr = state.v # m3/kg
            
            # Surface Production Well (2)
            P2 = P_whp # MPa
            h2 = h_pro[t] #kJ/kg
            state = therm(P=P2,h=h2)
            T2 = state.T # K
            s2 = state.s # kJ/kg-K
            x2 = state.x # steam quality
            v2 = state.v # m3/kg
            
            # Turbine Flow Stream (2s)
            state = therm(P=P2,x=1)
            P2s = state.P # MPa
            h2s = state.h #kJ/kg
            T2s = state.T # K
            s2s = state.s # kJ/kg-K
            x2s = state.x # steam quality
            v2s = state.v # m3/kg
            
            # Brine Flow Stream (2l)
            state = therm(P=P2,x=0)
            P2l = state.P # MPa
            h2l = state.h #kJ/kg
            T2l = state.T # K
            s2l = state.s # kJ/kg-K
            x2l = state.x # steam quality
            v2l = state.v # m3/kg
            
            # Turbine Outflow (3s)
            P3s = P_exh
            s3s = s2s
            state = therm(P=P3s,s=s3s)
            P3s = state.P # MPa
            h3s = state.h #kJ/kg
            T3s = state.T # K
            s3s = state.s # kJ/kg-K
            x3s = state.x # steam quality
            v3s = state.v # m3/kg
            
            # Condenser Outflow (4s)
            P4s = P3s
            T4s = T5
            state = therm(T=T4s,P=P4s)
            P4s = state.P # MPa
            h4s = state.h #kJ/kg
            T4s = state.T # K
            s4s = state.s # kJ/kg-K
            x4s = state.x # steam quality
            v4s = state.v # m3/kg
            
            # Turbine specific work
            w3s = h2s - h3s # kJ/kg
            
            # Pump specific Work
            w5s = v5*(P5-P4s)*10**3 # kJ/kg
            w5l = v5*(P5-P2l)*10**3 # kJ/kg
            
            # Mass flow rates
            mt = m_pro
            ms = mt*x2
            ml = mt*(1.0-x2)
            mi = m_inj
            Vt = (v5*mt)*(1000*60) # L/min
            
            # Pumping power
            Pump = 0.0
            if mi > mt:
                Pump = -1.0*(ms*w5s + ml*w5l)/effic + -1.0*(mi-mt)*w5s/effic #kW
            elif mi < ml:
                Pump = -1.0*(mi*w5l)/effic #kW
            else:
                Pump = -1.0*((mi-ml)*w5s + ml*w5l)/effic #kW
                    
            # Flash cycle power
            Flash = 0.0 #kW
            Flash = ms*np.max([0.0, w3s])*effic #kW
            
            # Binary cycle power
            Binary = 0.0
            # Outlet thermal state
            TBo = np.max([T5,51.85+273.15])
            PBo = np.min([P3s,P2])
            state = therm(T=TBo,P=PBo)
            PBo = state.P # MPa
            hBo = state.h #kJ/kg
            TBo = state.T # K
            # Binary Cycle Inlet from Turbine
            TBis = T3s
            PBis = P3s
            hBis = h3s
            # Binary Cycle Inlet from Brine
            TBil = np.min([T2l,T2])
            PBil = P2l
            hBil = np.min([h2l,h2])
            # Binary thermal-electric efficiency
            nBs = np.max([0.0, 0.0899*TBis - 25.95])/100.0
            nBl = np.max([0.0, 0.0899*TBil - 25.95])/100.0
            Binary = (ms*nBs*np.max([0.0, hBis-hBo]) + ml*nBl*np.max([0.0, hBil-hBo]))*effic #kW, power produced from binary cycle
            
            # Net power
            Net = 0.0
            Net = Flash + Binary + Pump
            
            # Record results
            Flash_Power += [Flash]
            Binary_Power += [Binary]
            Pump_Power += [Pump]
            Net_Power += [Net]
            
        Flash_Power = np.asarray(Flash_Power)
        Binary_Power = np.asarray(Binary_Power)
        Pump_Power = np.asarray(Pump_Power)
        Net_Power = np.asarray(Net_Power)       
    
        #print( details)
        if detail:
            print( '\n*** Rankine Cycle Thermal State Values ***')
            print( ("Inject (5): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T5,P5,h5,s5,x5,v5)))
            #print( ("Reserv (r,1): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(Tr,Pr,hr,sr,xr,vr)))
            print( ("Produc (2): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T2,P2,h2,s2,x2,v2)))
            print( ("Turbi (2s): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T2s,P2s,h2s,s2s,x2s,v2s)))
            print( ("Brine (2l): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T2l,P2l,h2l,s2l,x2l,v2l)))
            print( ("Exhau (3s): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T3s,P3s,h3s,s3s,x3s,v3s)))
            print( ("Conde (4s): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T4s,P4s,h4s,s4s,x4s,v4s)))
            print( '*** Binary Cycle Thermal State Values ***')
            print( ("Steam: Ti= %.2f; Pi= %.2f; hi= %.2f -> To= %.2f, Po= %.2f, ho= %.2f, n =%.3f" %(TBis,PBis,hBis,TBo,PBo,hBo,nBs)))
            print( ("Brine: Ti= %.2f; Pi= %.2f; hi= %.2f -> To= %.2f, Po= %.2f, ho= %.2f, n =%.3f" %(TBil,PBil,hBil,TBo,PBo,hBo,nBl)))
            print( '*** Power Output Estimation ***')            
            print( "Turbine Flow Rate = %.2f kg/s" %(ms))
            print( "Bypass Flow Rate = %.2f kg/s" %(ml))
            print( "Well Flow Rate = %.2f kg/s = %.2f L/min" %(mt, Vt))
            print( "Flash Power at %.2f kW" %(Flash))
            print( "Binary Power at %.2f kW" %(Binary))
            print( "Pumping Power at %.2f kW" %(Pump))
            print( "Net Power at %.2f kW" %(Net))
        
        #visualization
        if plots:
            #specific power vs enthalpy
            fig = pylab.figure(figsize=(8.0, 3.5), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
            ax1 = fig.add_subplot(111)
            ax1.plot(h_pro,np.asarray(Flash_Power)/np.asarray(m_pro),'.',color='m',linewidth=1.0,label='Flash')
            ax1.plot(h_pro,np.asarray(Binary_Power)/np.asarray(m_pro),'.',color='b',linewidth=1.0,label='Binary')
            ax1.plot(h_pro,-np.asarray(Pump_Power)/np.asarray(m_pro),'.',color='c',linewidth=1.0,label='Pump')
            ax1.plot(h_pro,np.asarray(Net_Power)/np.asarray(m_pro),':',color='r',linewidth=1.0,label='Net')
            ax1.set_xlabel('Production Enthalpy (kJ/kg)')
            ax1.set_ylabel('Power (kW/kg/s)')
            ax1.set_title('Specific Power vs Enthalpy')
            ax1.legend(loc='upper left', prop={'size':8}, ncol=1, numpoints=1)
            
            #raw power
            fig = pylab.figure(figsize=(8.0, 3.5), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
            ax1 = fig.add_subplot(111)
            ax1.plot(np.asarray(Flash_Power),'.',color='m',linewidth=1.0,label='Flash')
            ax1.plot(np.asarray(Binary_Power),'.',color='b',linewidth=1.0,label='Binary')
            ax1.plot(-np.asarray(Pump_Power),'.',color='c',linewidth=1.0,label='Pump')
            ax1.plot(np.asarray(Net_Power),':',color='r',linewidth=1.0,label='Net')
            ax1.set_xlabel('Sequence (time?)')
            ax1.set_ylabel('Power (kW)')
            ax1.set_title('Output and Input Power')
            ax1.legend(loc='upper right', prop={'size':8}, ncol=1, numpoints=1)        
        
        #returns
        return Flash_Power, Binary_Power, Pump_Power, Net_Power
    
    # ************************************************************************
    # optimization objective function (with $!!!), to be normalized to 2021 studies
    # ************************************************************************
    def get_economics(time, #yr - power data's timeseries
                      depth, #m - resource depth
                      lateral, #m - lateral length
                      Net_Power, #kW - net produced electric power including pumping losses
                      Max_Quake, #Mw - earthquake magnitude
                      well_count, #wells - number of wells
                      interest = 0.04, #standard inflation rate, 4% rule
                      sales_kWh = 0.1372, #$/kWh - customer electricity retail price                  
                      drill_m = 2763.06, #$/m - Lowry et al, 2017 large diameter well baseline
                      pad_fixed = 590e3, #$ Lowry et al, 2017 large diameter well baseline
                      plant_kWe = 2025.65, #$/kWe simplified from GETEM model 
                      explore_m = 2683.41, #$/m simplified from GETEM model
                      oper_kWh = 0.03648, #$/kWh simplified from GETEM model
                      quake_coef = 2e-4, #$/Mw for $300M Mw 5.5 quake Pohang (Westaway, 2021) & $17.2B Mw 6.3 quake Christchurch (Swiss Re)
                      quake_exp = 5.0, #$/Mw for $300M Mw 5.5 quake Pohang (Westaway, 2021) & $17.2B Mw 6.3 quake Christchurch (Swiss Re)
                      detail=False, #print cycle infomation
                      plots=False): #plot results
        #eliminate periods of negative net power production
        Pout_NN = Net_Power+0.0
        Pout_NN[Pout_NN < 0.0] = 0.0
        NPsum = np.sum(Pout_NN)
        #average kilowatt-hour
        dt = time[1]-time[0]
        life = time[-1]-time[0]
        #power profit
        P = (sales_kWh-oper_kWh)*NPsum*dt*24.0*365.2425 #!!! should this include discount rate? -- probably not
        #capital costs
        C = 0.0
        C += drill_m*(depth*well_count + lateral*well_count)
        C += pad_fixed
        C += plant_kWe*NPsum*dt/life #!!! should this be based on peak power costs? -- probably not
        C += explore_m*depth
        #quake cost
        Q = quake_coef * np.exp(Max_Quake*quake_exp)
        #net present value
        NPV = P - C - Q
        return NPV, P, C, Q

    #recalculate NPV
    effic = 0.85
    for i in range(0,len(data)):
        Flash_Power, Binary_Power, Pump_Power, Net_Power = get_power(data['minj'][i],data['Tinj'][i]+273.15,data['pinj'][i],-data['mpro'][i],hpro[i],1.0,0.1,effic)
        NPV, P, C, Q = get_economics(ts,data['ResDepth'][i],data['w_length'][i],Net_Power,data['max_quake'][i],data['w_count'][i])
        data['NPV'][i] = NPV
        data['profit_sales'][i] = P
        data['cost_capital'][i] = C
        data['risk_quakes'][i] = Q
            
    #Generation 2
    if True:
        #NPV vs flow rate and spacing
        fig = pylab.figure(figsize=(15.0,7.5),dpi=100)
        ax1 = fig.add_subplot(111)
        sc = ax1.scatter(x=data['w_spacing'],y=data['Qinj'],c=data['NPV'],vmin=-50e6,vmax=50e6,cmap=plt.cm.get_cmap('rainbow_r'))
        plt.colorbar(mappable=sc,label='Net Present Value ($)',orientation='vertical')
        ax1.set_xlabel('Well Spacing (m)',fontsize=10)
        ax1.set_ylabel('Injection Rate (m3/s)',fontsize=10)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.savefig('NPV_v_S_v_Q_optQinj_85.png', format='png')

    #Generation 2
    if True:
        # pie chart: NPV
        sizes = []
        labels = ['>$50M', '%$-50M to %$0M','%$0M to %$50M', '<-$50M']
        keep = np.zeros(4,dtype=int)
        for j in range(0,len(data)):
            if data['NPV'][j] < -50.0e6:
                keep[3] += 1
            elif data['NPV'][j] < 0.0:
                keep[1] += 1
            elif data['NPV'][j] < 50.0e6:
                keep[2] += 1
            else:
                keep[0] += 1
            sizes = 1.0*keep/np.sum(keep)
        explode = (0,0,0,0)
        fig = pylab.figure(figsize=(10,5),dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=90)
        ax1.axis('equal')
        ax1.set_title('Net Present Value (%i samples)' %(np.sum(keep)))
        plt.tight_layout()
        plt.savefig('npv_optQinj_85.png', format='png')
        pylab.close() 
    
    #P10 statistics
    if True:
        #cumulative NPV function
        npvs = data['NPV']+0.0
        npvs = np.sort(npvs)
        nums = len(npvs)
        #other stuff
        pows = data['profit_sales']+0.0
        pows = np.sort(pows)
        cost = data['cost_capital']+0.0
        cost = np.sort(cost)
        quak = data['risk_quakes']+0.0
        quak = np.sort(quak)
        
        #collect percentiles
        out = []
        pcts = [0,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99,100]
        for p in pcts:
            out += [['npv%i' %(p), npvs[np.min([nums-1,int(1.0*nums*p/100)])]]]
            out += [['p%i' %(p), pows[np.min([nums-1,int(1.0*nums*p/100)])]]]
            out += [['c%i' %(p), cost[np.min([nums-1,int(1.0*nums*p/100)])]]]
            out += [['q%i' %(p), quak[np.min([nums-1,int(1.0*nums*p/100)])]]]
        
        #save to file
        fname = "keys_85_" + filename
        out = list(map(list,zip(*out)))
        head = out[0][0]
        for i in range(1,len(out[0])):
            head = head + ',' + out[0][i]
        sdata = '%i' %(out[1][0])
        for i in range(1,len(out[1])):          
            if not out[1][i]:
                sdata = sdata + ',0.0' # ',nan'
            else:
                sdata = sdata + ',%.5e' %(out[1][i])
        try:
            with open(fname,'r') as f:
                test = f.readline()
            f.close()
            if test != '':
                with open(fname,'a') as f:
                    f.write(sdata + '\n')
                f.close()
            else:
                with open(fname,'a') as f:
                    f.write(head + '\n')
                    f.write(sdata + '\n')
                f.close()
        except:
            with open(fname,'a') as f:
                f.write(head + '\n')
                f.write(sdata + '\n')
            f.close()
    
pylab.show()

