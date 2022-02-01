# ****************************************************************************
#### EGS collab example
# ****************************************************************************

# ****************************************************************************
#### standard imports
# ****************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import GeoDT as gt
import pylab
import copy
import math
deg = gt.deg
MPa = gt.MPa
GPa = gt.GPa
yr = gt.yr
cP = gt.cP
mD = gt.mD
mLmin = 1.66667e-8 #m3/s
gal = 1.0/264.172 #m3

# ****************************************************************************
#### model setup
# ****************************************************************************        

#full randomizer
for i in range(0,100):
    #create model object
    geom = []
    geom = gt.mesh()
    
#        #rock properties
    geom.rock.size = 140.0 #m #!!!
    geom.rock.ResDepth = np.random.uniform(1250.0,1250.0) #6000.0 # m #!!!
    geom.rock.ResGradient = np.random.uniform(34.0,36.0) #50.0 #56.70 # C/km; average = 25 C/km #!!!
    geom.rock.ResRho = np.random.uniform(2925.0,3040.0) #2700.0 # kg/m3 #!!!
    geom.rock.ResKt = np.random.uniform(2.55,3.81) #2.5 # W/m-K #!!!
    geom.rock.ResSv = np.random.uniform(1900.0,2200.0) #2063.0 # kJ/m3-K
    geom.rock.AmbTempC = np.random.uniform(20.0,20.0) #25.0 # C #!!!
    geom.rock.AmbPres = 0.101 #Example: 0.01 MPa #Atmospheric: 0.101 # MPa
    geom.rock.ResE = np.random.uniform(89.0,110.0)*GPa #50.0*GPa #!!!
    geom.rock.Resv = np.random.uniform(0.17,0.28) #0.3 #!!!
    geom.rock.Ks3 = 0.26197 #np.random.uniform(0.5,0.5) #0.5 #!!!
    geom.rock.Ks2 = 1.05421 #geom.rock.Ks3 + np.random.uniform(0.4,0.6) # 0.75 #!!!
    geom.rock.s3Azn = 14.4*deg #!!!
    geom.rock.s3AznVar = 5.0*deg #!!!
    geom.rock.s3Dip = 27.0*deg #!!!
    geom.rock.s3DipVar = 5.0*deg #!!!
    #fracture orientation parameters #[i,:] set, [0,0:2] min, max --or-- nom, std
#        geom.rock.fNum = np.asarray([int(np.random.uniform(0,30)),
#                                int(np.random.uniform(0,30)),
#                                int(np.random.uniform(0,30))],dtype=int) #count
#        r1 = np.random.uniform(50.0,800.0)
#        r2 = np.random.uniform(50.0,800.0)
#        r3 = np.random.uniform(50.0,800.0)
#        geom.rock.fDia = np.asarray([[r1,r1+np.random.uniform(100.0,800.0)],
#                                [r2,r2+np.random.uniform(100.0,800.0)],
#                                [r3,r3+np.random.uniform(100.0,800.0)]],dtype=float) #m
#        geom.rock.fStr = np.asarray([[np.random.uniform(0.0,360.0)*deg,np.random.uniform(0.0,90.0)*deg],
#                                [np.random.uniform(0.0,360.0)*deg,np.random.uniform(0.0,90.0)*deg],
#                                [np.random.uniform(0.0,360.0)*deg,np.random.uniform(0.0,90.0)*deg]],dtype=float) #m
#        geom.rock.fDip = np.asarray([[np.random.uniform(0.0,90.0)*deg,np.random.uniform(0.0,45.0)*deg],
#                                [np.random.uniform(0.0,90.0)*deg,np.random.uniform(0.0,45.0)*deg],
#                                [np.random.uniform(0.0,90.0)*deg,np.random.uniform(0.0,45.0)*deg]],dtype=float) #m
    geom.rock.fNum = np.asarray([int(np.random.uniform(0,30)),
                            int(np.random.uniform(0,30)),
                            int(np.random.uniform(0,30))],dtype=int) #count
    geom.rock.fDia = np.asarray([[50.0,180.0],
                            [50.0,180.0],
                            [50.0,180.0]],dtype=float) #m
    #EGS Collab #!!!
    geom.rock.fStr = np.asarray([[15.0*deg,7.0*deg],
                            [260.0*deg,7.0*deg],
                            [120.0*deg,7.0*deg,]],dtype=float) #m
    geom.rock.fDip = np.asarray([[35.0*deg,7.0*deg],
                            [69.0*deg,7.0*deg,],
                            [35.0*deg,7.0*deg]],dtype=float) #m
    #fracture hydraulic parameters
    geom.rock.gamma = np.asarray([10.0**-3.0,10.0**-2.0,10.0**-1.2])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.000,0.200,0.800])
    geom.rock.b = np.asarray([0.999,1.0,1.001])
    geom.rock.N = np.asarray([0.0,0.6,2.0])
    geom.rock.alpha = np.asarray([2.0e-9,2.9e-8,10.0e-8])
    geom.rock.bh = np.asarray([0.00000001,0.00005,0.0001]) #np.asarray([0.00005,0.00010,0.00020]) #!!!
#     #fracture hydraulic parameters
# #        r = np.random.exponential(scale=0.25,size=2)
# #        r[r>1.0] = 1.0
# #        r[r<0] = 0.0
# #        r = r*(0.100/MPa-0.001/MPa)+0.001/MPa   
# #        u1 = -np.min(r)
# #        u2 = -np.max(r)
# #        u3 = 0.5*(u1+u2)
# #        geom.rock.alpha = np.asarray([u1,u3,u2])
#     geom.rock.alpha = np.asarray([-0.028/MPa,-0.028/MPa,-0.028/MPa])
    
# #        r = np.random.exponential(scale=0.25,size=2)
# #        r[r>1.0] = 1.0
# #        r[r<0] = 0.0
# #        r = r*(0.1-0.001)+0.001   
# #        u1 = np.min(r)
# #        u2 = np.max(r)
# #        u3 = 0.5*(u1+u2)
# #        geom.rock.gamma = np.asarray([u1,u3,u2])
#     geom.rock.gamma = np.asarray([0.01,0.01,0.01])
    
#     geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    
# #        r = np.random.exponential(scale=0.25,size=2)
# #        r[r>1.0] = 1.0
# #        r[r<0] = 0.0
# #        r = r*(0.2-0.012)+0.012   
# #        u1 = np.min(r)
# #        u2 = np.max(r)
# #        u3 = 0.5*(u1+u2)
# #        geom.rock.a = np.asarray([u1,u3,u2])
#     geom.rock.a = np.asarray([0.05,0.05,0.05])
    
# #        u1 = np.random.uniform(0.7,0.9)
# #        u2 = np.random.uniform(0.7,0.9)
# #        u3 = 0.5*(u1+u2)
# #        geom.rock.b = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
#     geom.rock.b = np.asarray([0.8,0.8,0.8])
    
# #        u1 = np.random.uniform(0.2,1.2)
# #        u2 = np.random.uniform(0.2,1.2)
# #        u3 = 0.5*(u1+u2)
# #        geom.rock.N = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
#     geom.rock.N = np.asarray([0.2,0.5,1.2])
    
# #        u1 = np.random.uniform(0.00005,0.00015)
# #        u2 = np.random.uniform(0.00005,0.00015)
# #        u3 = 0.5*(u1+u2)
# #        geom.rock.bh = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
#     geom.rock.bh = np.asarray([0.00005,0.0001,0.003])
    
    geom.rock.bh_min = 0.00000005 #m #!!!
    geom.rock.bh_max = 0.0001 #0.02000 #m #!!!
#        geom.rock.bh_bound = np.random.uniform(0.001,0.005)
    geom.rock.bh_bound = np.random.uniform(0.00000005,0.0001) #!!!
    geom.rock.f_roughness = np.random.uniform(0.25,1.0) #0.8
    #well parameters
    geom.rock.w_count = 4 #2 #wells
    geom.rock.w_spacing = 30.0 #np.random.uniform(100.0,800.0) #300.0 #m
    geom.rock.w_length = 60.0 #1500.0 #800.0 #m
    geom.rock.w_azimuth = 60.0*deg #geom.rock.s3Azn + np.random.uniform(-15.0,15.0)*deg #rad
    geom.rock.w_dip = 20.0*deg #geom.rock.s3Dip + np.random.uniform(-15.0,15.0)*deg #rad
    geom.rock.w_proportion = 0.8 #m/m
    geom.rock.w_phase = 15.0*deg #rad
    geom.rock.w_toe = -35.0*deg #rad
    geom.rock.w_skew = 0.0*deg #rad
    geom.rock.w_intervals = 2 #int(np.random.uniform(1,3)) #3 #breaks in well length
    geom.rock.ra = 0.1*0.0191 #limited by a_max, actual is 0.096 m #!!!
    geom.rock.rgh = 80.0 #!!!
    #cement properties
    geom.rock.CemKt = 2.0 # W/m-K
    geom.rock.CemSv = 2000.0 # kJ/m3-K
    #thermal-electric power parameters
    geom.rock.GenEfficiency = 0.85 # kWe/kWt
    geom.rock.LifeSpan = 1.0*yr/2 #years #!!!
    geom.rock.TimeSteps = 41 #steps
    geom.rock.p_whp = 1.0*MPa #Pa
    geom.rock.Tinj = 10.0 #95.0 #C #!!!
    geom.rock.H_ConvCoef = 3.0 #kW/m2-K
    geom.rock.dT0 = 1.0 #K #!!!
    geom.rock.dE0 = 50.0 #kJ/m2 #!!!
    #water base parameters
    geom.rock.PoreRho = 980.0 #kg/m3 starting guess
    geom.rock.Poremu = 0.9*cP #Pa-s
    geom.rock.Porek = 0.1*mD #m2
    geom.rock.Frack = 100.0*mD #m2
    #stimulation parameters
    if geom.rock.w_intervals == 1:
        geom.rock.perf = int(np.random.uniform(1,1))
    else:
        geom.rock.perf = 1
    geom.rock.r_perf = 5.0 #m #!!!
    geom.rock.sand = 0.3 #sand ratio in frac fluid
    geom.rock.leakoff = 0.0 #Carter leakoff
#        geom.rock.dPp = -1.0*np.random.uniform(1.0,10.0)*MPa #-2.0*MPa #production well pressure drawdown
    geom.rock.dPp = -2.0*MPa #production well pressure drawdown
    geom.rock.dPi = 0.1*MPa #!!!
    geom.rock.stim_limit = 5
#    geom.rock.Qinj = 0.01 #m3/s
    # geom.rock.Qstim = 0.01 #0.08 #m3/s
    # geom.rock.Vstim = 1000.0 #100000.0 #m3
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
#        u1 = np.random.uniform(20.0,55.0)*deg
#        u2 = np.random.uniform(20.0,55.0)*deg
#        u3 = 0.5*(u1+u2)
#        geom.rock.phi = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])]) #rad
    geom.rock.phi = np.asarray([20.0*deg,35.0*deg,45.0*deg]) #rad #!!!
#        u1 = np.random.uniform(5.0,20.0)*MPa
#        u2 = np.random.uniform(5.0,20.0)*MPa
#        u3 = 0.5*(u1+u2)
#        geom.rock.mcc = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])]) #Pa
    geom.rock.mcc = np.asarray([2.0,10.0,15.0])*MPa #Pa #!!!
    geom.rock.hfmcc = np.random.uniform(0.0,0.2)*MPa #0.1*MPa #!!!
    geom.rock.hfphi = np.random.uniform(15.0,35.0)*deg #30.0*deg #!!!
    #**********************************

    #recalculate base parameters
    geom.rock.re_init()
    
    #generate domain
    geom.gen_domain()

    #generate natural fractures
    geom.gen_joint_sets()
    
    # ****************************************************************************
    #### placed fractures
    # ****************************************************************************
    data = np.recfromcsv('Well Geometry Info.csv',delimiter=',',filling_values=np.nan,deletechars='()',case_sensitive=True,names=True)
    dia = 5.0
    for i in range(0,len(data)):
        if data['type'][i] == 1:
            c0 = [data['x_m'][i],data['y_m'][i],data['z_m'][i]]
            dia = dia
            strike = data['azn_deg'][i]*deg
            dip = data['dip_deg'][i]*deg
            geom.gen_fixfrac(False,c0,dia,strike,dip)
            
    # ****************************************************************************
    #### common well geometry
    # ****************************************************************************   
    
    #generate wells
    # wells = []
    # geom.gen_wells(True,wells)
    # well geometry using gyro log for E2-TC and centered on kickoff point of E2-TC, other wells based on original design Jan 2021
    #gt.line(x0=0.0,y0=0.0,z0=0.0,length=1.0,azn=0.0*deg,dip=0.0*deg,w_type='pipe',dia=0.0254*3.0,rough=80.0)
    wells = []
    
    #monitoring    
    wells += [gt.line(5.0282856,51.9184128,-0.931164,49.9872,0,1.570796327,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(8.019288,53.2199088,0.2785872,10.668,0.788888822,0.093724181,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(9.0034872,51.1996944,-0.3691128,59.436,1.752310569,0.67718775,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(9.1452192,51.3560568,0.001524,59.436,1.761037215,0.151843645,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(-1.00584,41.0135832,-0.3447288,54.864,2.138028334,0.616101226,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(-1.0107168,41.192196,0.185928,54.864,2.038544566,0.02443461,'producer',geom.rock.ra, geom.rock.rgh)]
    
    #injector
    #wells += [gt.line(0,0,0,77.10353424,0.832584549,0.257981302,'injector',geom.rock.ra, geom.rock.rgh)]
    
    #producers
    wells += [gt.line(-0.6257544,0.4255008,-0.0356616,76.2,0.740019603,0.230383461,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(-0.7824216,0.4474464,-0.3249168,76.2,0.841248699,0.390953752,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(0.1679448,-0.4300728,0.0917448,76.2,0.900589894,0.132645023,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(0.2505456,-0.760476,-0.1063752,80.772,1.007054978,0.287979327,'producer',geom.rock.ra, geom.rock.rgh)]
    
    #drift
    wells += [gt.line(-4.3290744,-70.1844672,-0.077724,155.7528,0.019024089,0,'producer',geom.rock.ra, geom.rock.rgh)]
    wells += [gt.line(-2.1954744,51.7355328,-0.077724,11.5824,1.584060829,0,'producer',geom.rock.ra, geom.rock.rgh)]
    #wells += [gt.line(-1.9211544,85.5073728,-0.077724,77.724,6.119124357,0,'producer',geom.rock.ra, geom.rock.rgh)]
    #wells += [gt.line(-8.5657944,125.4056928,-0.077724,25.908,0.944223125,0,'producer',geom.rock.ra, geom.rock.rgh)]
    #wells += [gt.line(12.4958856,140.6456928,-0.077724,29.8704,5.246459731,0,'producer',geom.rock.ra, geom.rock.rgh)]
    
    #split injector in different locations
    for s in range(0,5):
        #copy natural fracture geometry
        comm = []
        comm = copy.deepcopy(geom)
        wells2 = []
        wells2 = copy.deepcopy(wells)
        
        #select interval        
        zone_leg = 1.5 #packer interval length
        comm.rock.sand = zone_leg #!!!
        rati_leg = np.random.uniform(0.3,0.95) #interval center depth
        comm.rock.leakoff = rati_leg #!!!
        azn = 0.832584549
        dip = 0.257981302
        leg = 77.10353424
        vAxi = np.asarray([math.sin(azn)*math.cos(-dip), math.cos(azn)*math.cos(-dip), math.sin(-dip)])
        x0 = np.asarray([0.0, 0.0, 0.0])
        x1 = x0 + vAxi*(rati_leg*leg - 0.5*zone_leg)
        x2 = x0 + vAxi*(rati_leg*leg + 1.0*zone_leg)
        wells2 += [gt.line(x0[0],x0[1],x0[2],rati_leg*leg - 1.0*zone_leg,0.832584549,0.257981302,'producer',comm.rock.ra, comm.rock.rgh)]
        wells2 += [gt.line(x1[0],x1[1],x1[2],1.0*zone_leg,0.832584549,0.257981302,'injector',comm.rock.ra, comm.rock.rgh)]
        wells2 += [gt.line(x2[0],x2[1],x2[2],leg-(rati_leg*leg + 1.0*zone_leg),0.832584549,0.257981302,'producer',comm.rock.ra, comm.rock.rgh)]

        #install
        comm.wells = wells2
        
        #stimulate
        comm.rock.Qstim = np.random.uniform(500.0*mLmin, 10000.0*mLmin) #0.08 #m3/s
        comm.rock.Vstim = np.random.uniform(100.0*gal, 10000.0*gal) #100000.0 #m3
        comm.dyn_stim(Vinj=comm.rock.Vstim,Qinj=comm.rock.Qstim,target=[],
                      visuals=False,fname='stim')
        
        #test multiple randomly selected flow rates
        rates = np.random.uniform(100.0*mLmin,10000.0*mLmin,5) 
        for r in rates:
            #copy base parameter set
            base = []
            base = copy.deepcopy(comm)
            
            #set rate
            base.rock.Qinj = r
            base.rock.re_init()
        
            #random identifier (statistically should be unique)
            pin = np.random.randint(100000000,999999999,1)[0]
            
            try:
    #            if True:
    #                #single rate long term flow
    #                base.dyn_flow(target=[],visuals=False,fname='run_%i' %(pin))
                
    #                #stim then flow
    #                base.stim_and_flow(target=[],visuals=False,fname='run_%i' %(pin))
                
                #Solve production
                base.dyn_stim(Vinj=base.rock.Vinj,Qinj=base.rock.Qinj,target=[],
                              visuals=False,fname='run_%i' %(pin))
                
                #calculate heat transfer
                base.get_heat(plot=True)
                plt.savefig('plt_%i.png' %(pin), format='png')
                plt.close()
            except:
                print( 'solver failure!')
                
            #show flow model
            base.build_vtk(fname='fin_%i' %(pin))
            
            if False: #3D temperature visual
                base.build_pts(spacing=50.0,fname='fin_%i' %(pin))
            
            #save primary inputs and outputs
            x = base.save('inputs_results_collabE2.txt',pin)


#show plots
pylab.show()