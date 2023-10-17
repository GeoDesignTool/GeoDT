# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:34:22 2023

@author: 299494
"""

import GeoDT as gt
import pylab
# import numpy as np

try:
    a = gt.visualization('inputs_results.txt')
except:
    a = gt.visualization('inputs_results.csv')

#### filters
if True:
    #filter out failed models - incorrect flow solution
    a.data['qinj']=a.data['qinj']/2
    a.data['qpro']=a.data['qpro']/2
    a.multifilter('qinj','Qinj',[0.9,1.1]) #normal
    
    #filter out failed models - injection pressure out of limits
    a.multifilter('pinj','s3',[0.00,0.95e-6])
    
#### Boxplot
if False:
    a.boxplot(criteria=[['Pavg',[1e3],'above'],
                        ['recovery',[0.90,1.11],'between'],
                        ['breakthrough',[0.5,9.0],'between'],
                        ['max_quake',[3.0],'below']],
                column=['ResDepth','qinj','w_spacing','perf_clusters','w_intervals','kf','perf_dia','dPp','pinj','qpro'])

#### 3D scatterplots 
if True:
    #scatter_3D(x,y,c,name=['x','y','c','save'],cmap='rainbow_r',xlog=False,ylog=False,vrange=[],view=False):
    #spacing, flow, power
    a.scatter_3D(a.data['w_spacing'],a.data['qinj'],a.Pavg,
               ['Well Spacing (m)','Injection Rate (m3/s)','Average Net Power per Year (kW)','s_qinj_APY'],'rainbow_r',False,True,[]) 
    
    #spacing, flow, NPV
    a.scatter_3D(a.data['w_spacing'],a.data['qinj'],a.data['NPV'],
                 ['Well Spacing (m)','Injection Rate (m3/s)','Net Present Value ($)','s_qinj_NPV'],'rainbow_r',False,True,[-200e6,200e6]) 
    
    #spacing, flow, $/Mw-net
    a.scatter_3D(a.data['w_spacing'],a.data['qinj'],a.data['CpM_net'],
                ['Well Spacing (m)','Injection Rate (m3/s)','Cost per MWe Net ($/MW)','s_qinj_CpMnet'],'rainbow',False,True,[0.0,200.0])
    
    #flow, power, npv
    a.scatter_3D(a.data['qinj'],a.Pavg,a.data['NPV'],
               ['Injection Rate (m3/s)','Average Net Power per Year (kW)','Net Present Value ($)','qinj_APY_NPV'],'rainbow_r',True,False,[-200e6,200e6]) 

    #flow, power, $/Mw-net
    a.scatter_3D(a.data['qinj'],a.Pavg,a.data['CpM_net'],
                ['Injection Rate (m3/s)','Average Net Power per Year (kW)','Cost per MWe Net ($/MW)','qinj_APY_CpMnet'],'rainbow',True,False,[0.0,200.0])

    #flow, power, $/Mw-tot
    a.scatter_3D(a.data['qinj'],a.Pavg,a.data['CpM_gro'],
                ['Injection Rate (m3/s)','Average Net Power per Year (kW)','Cost per MWe Gross ($/MW)','qinj_APY_CpMtot'],'rainbow',True,False,[0.0,200.0])

    #flow, power, $/Mw-themal
    a.scatter_3D(a.data['qinj'],a.Pavg,a.data['CpM_the'],
                ['Injection Rate (m3/s)','Average Net Power per Year (kW)','Cost per MW Thermal ($/MW)','qinj_APY_CpMther'],'rainbow',True,False,[0.0,200.0])
    
    #flow, Kr, power
    a.scatter_3D(a.data['qinj'],a.data['ResKt'],a.Pavg,
               ['Injection Rate (m3/s)','Thermal Conductivity (w/m2)','Average Net Power per Year (kW)','qinj_Kr_APY'],'rainbow_r',True,False,[]) 
    
    #flow, BHT, power
    a.scatter_3D(a.data['qinj'],a.data['BH_T'],a.Pavg,
               ['Injection Rate (m3/s)','Bottomhole Temperature (K)','Average Net Power per Year (kW)','qinj_BHT_APY'],'rainbow_r',True,False,[])
    
    #flow, kf, power
    a.scatter_3D(a.data['qinj'],a.data['kf']/gt.um2cm,a.Pavg,
                ['Injection Rate (m3/s)','Fracture Conductivity (um2-cm)','Average Net Power per Year (kW)','qinj_kf_APY'],'rainbow_r',True,False,[])

    #flow, perf_dia, power
    a.scatter_3D(a.data['qinj'],a.data['perf_dia'],a.Pavg,
                ['Injection Rate (m3/s)','Perforation Diameter (m)','Average Net Power per Year (kW)','qinj_perf_dia_APY'],'rainbow_r',True,False,[])

pylab.show()