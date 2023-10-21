# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:34:22 2023

@author: 299494
"""

import GeoDT as gt
import pylab
import numpy as np

if True: #default import method
    try:
        a = gt.visualization('inputs_results.txt')
    except:
        a = gt.visualization('inputs_results.csv')
else: #two-batch models
    b = gt.visualization('inputs_results.txt')
    a = gt.visualization('inputs_results.csv')
    a.orig['qinj'] = a.orig['qinj']/2 #required for EGS wells in bugged version of GeoDT
    a.orig['qpro'] = a.orig['qpro']/2 #required for EGS wells in bugged version of GeoDT
    a.orig = np.append(a.orig,b.orig,axis=0)
    a.reset()

#### filters
if True:
    if False: #comparatively filter out failed models - incorrect flow solution
        a.multifilter('qinj','Qinj',[0.9,1.1])
    else: #absolutely filter out failed models - excess flow
        a.multifilter('qinj',None,[0.001,0.8])
    if False: #filter out failed models - injection pressure out of limits
        a.multifilter('pinj','s3',[0.00,0.95e-6])
    else: #impossible injection pressures
        a.multifilter('pinj',None,[0.00,140.0])
    
#### Boxplot
if False:
    a.boxplot(criteria=[['Pavg',[1e3],'above'],
                        ['recovery',[0.90,1.11],'between'],
                        ['breakthrough',[0.5,9.0],'between'],
                        ['max_quake',[3.0],'below']],
                column=['ResDepth','qinj','w_spacing','perf_clusters','w_intervals','kf','perf_dia','dPp','pinj','qpro'])

#### 3D scatterplots 
def scatter(suffix=''):
    #scatter_3D(x,y,c,name=['x','y','c','save'],cmap='rainbow_r',xlog=False,ylog=False,vrange=[],view=False):
    #spacing, flow, power
    a.scatter_3D(a.data['w_spacing'],a.data['qinj'],a.Pavg,
               ['Well Spacing (m)','Injection Rate (m3/s)','Average Net Power per Year (kW)','s_qinj_APY'+suffix],'rainbow_r',False,True,[]) 
    
    #spacing, flow, NPV
    a.scatter_3D(a.data['w_spacing'],a.data['qinj'],a.data['NPV'],
                 ['Well Spacing (m)','Injection Rate (m3/s)','Net Present Value ($)','s_qinj_NPV'+suffix],'rainbow_r',False,True,[-200e6,200e6]) 
    
    #spacing, flow, $/Mw-net
    a.scatter_3D(a.data['w_spacing'],a.data['qinj'],a.data['CpM_net'],
                ['Well Spacing (m)','Injection Rate (m3/s)','Cost per MWe Net ($/MW)','s_qinj_CpMnet'+suffix],'rainbow',False,True,[0.0,200.0])
    
    #flow, power, npv
    a.scatter_3D(a.data['qinj'],a.Pavg,a.data['NPV'],
               ['Injection Rate (m3/s)','Average Net Power per Year (kW)','Net Present Value ($)','qinj_APY_NPV'+suffix],'rainbow_r',True,False,[-200e6,200e6]) 

    #flow, power, $/Mw-net
    a.scatter_3D(a.data['qinj'],a.Pavg,a.data['CpM_net'],
                ['Injection Rate (m3/s)','Average Net Power per Year (kW)','Cost per MWe Net ($/MW)','qinj_APY_CpMnet'+suffix],'rainbow',True,False,[0.0,200.0])

    #flow, power, $/Mw-tot
    a.scatter_3D(a.data['qinj'],a.Pavg,a.data['CpM_gro'],
                ['Injection Rate (m3/s)','Average Net Power per Year (kW)','Cost per MWe Gross ($/MW)','qinj_APY_CpMtot'+suffix],'rainbow',True,False,[0.0,200.0])

    #flow, power, $/Mw-themal
    a.scatter_3D(a.data['qinj'],a.Pavg,a.data['CpM_the'],
                ['Injection Rate (m3/s)','Average Net Power per Year (kW)','Cost per MW Thermal ($/MW)','qinj_APY_CpMther'+suffix],'rainbow',True,False,[0.0,200.0])
    
    # #flow, Kr, power
    # a.scatter_3D(a.data['qinj'],a.data['ResKt'],a.Pavg,
    #            ['Injection Rate (m3/s)','Thermal Conductivity (w/m2)','Average Net Power per Year (kW)','qinj_Kr_APY'+suffix],'rainbow_r',True,False,[]) 
    
    # #flow, BHT, power
    # a.scatter_3D(a.data['qinj'],a.data['BH_T'],a.Pavg,
    #            ['Injection Rate (m3/s)','Bottomhole Temperature (K)','Average Net Power per Year (kW)','qinj_BHT_APY'+suffix],'rainbow_r',True,False,[])
    
    # #flow, kf, power
    # a.scatter_3D(a.data['qinj'],a.data['kf']/gt.um2cm,a.Pavg,
    #             ['Injection Rate (m3/s)','Fracture Conductivity (um2-cm)','Average Net Power per Year (kW)','qinj_kf_APY'+suffix],'rainbow_r',True,False,[])

    # #flow, perf_dia, power
    # a.scatter_3D(a.data['qinj'],a.data['perf_dia'],a.Pavg,
    #             ['Injection Rate (m3/s)','Perforation Diameter (m)','Average Net Power per Year (kW)','qinj_perf_dia_APY'+suffix],'rainbow_r',True,False,[])
scatter('_full')

a.multifilter('qinj',None,[0.02,0.20])
a.multifilter('w_spacing',None,[400,600])

scatter('_goal')
a.percentiles('percentiles')

#pylab.show()