# -*- coding: utf-8 -*-

# ****************************************************************************
# Functions for analyzing and visualizing results from GeoDT
# Author: Luke P. Frash
#
# Notation:
#  !!! for code needing to be updated (e.g, limititations or work in progress)
#  *** for breaks betweeen key sections
# ****************************************************************************

# ****************************************************************************
#### libraries
# ****************************************************************************
import numpy as np
#from scipy.linalg import solve
#from scipy.stats import lognorm
import pylab
import math
from iapws import IAPWS97 as therm
#from libs import SimpleGeometry as sg
#from scipy import stats
#import sys
# import matplotlib.pyplot as plt
# import copy

# ****************************************************************************
#### unit conversions
# ****************************************************************************
# Unit conversions for convenience
lpm=(0.1*0.1*0.1)/(60.0)
ft=12*25.4e-3#m
m=(1.0/ft)#ft
deg=1.0*math.pi/180.0
#rad=1.0/deg
gal=3.785*0.1*0.1*0.1#m^3=3.785 l
gpm=gal/60.0
liter=0.1*0.1*0.1
lps=liter/1.0# = liters per second
cP=10.0**-3#Pa-s
g=9.81#m/s2
MPa=10.0**6.0#Pa
GPa=10.0**9.0#Pa
darcy=9.869233*10**-13#m2
mD=darcy*10.0**3.0#m2
yr=365.2425*24.0*60.0*60.0#s
pi=math.pi

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

# ****************************************************************************
#### main program
# ****************************************************************************
if True: #main program
    hpro = np.flip(np.asarray([300,350,400,450,500,550,600,650,700,750,800,850,869.25,900,950,1000,1050,1100,1150,1200]))
    hpro = np.flip(np.asarray([800,850,869.25,900,950,1000,1050,1100,1150,1200]))
    ts = np.linspace(0,20,len(hpro))

    F, B, P, N = get_power(34.27, 346.0, 20.37, 34.27, 
                           hpro, 
                           1.0, 0.10, 0.85, True, True)
    
    NPV = get_economics(ts,4000.0,1000.0,N,3.5)
    print('\n*** NPV = %.2f' %(NPV))
    
    pylab.show()