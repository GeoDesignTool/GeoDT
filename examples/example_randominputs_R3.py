# ****************************************************************************
#### randomized inputs w/ scaled flow
# ****************************************************************************

# ****************************************************************************
#### standard imports
# ****************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import GeoDT as gt
import pylab
import copy
deg = gt.deg
MPa = gt.MPa
GPa = gt.GPa
yr = gt.yr
cP = gt.cP
mD = gt.mD

# ****************************************************************************
#### model setup
# ****************************************************************************
#full randomizer
for i in range(0,1):
    #create model object
    geom = []
    geom = gt.mesh()

    #rock properties
    geom.rock.size = 1500.0 #m
    geom.rock.ResDepth = np.random.uniform(7000.0,8000.0) #6000.0 # m
    geom.rock.ResGradient = np.random.uniform(45.0,50.0) #50.0 #56.70 # C/km; average = 25 C/km
    geom.rock.ResRho = np.random.uniform(2650.0,2750.0) #2700.0 # kg/m3
    geom.rock.ResKt = np.random.uniform(2.4,2.6) #2.5 # W/m-K
    geom.rock.ResSv = np.random.uniform(1950.0,2050.0) #2063.0 # kJ/m3-K
    geom.rock.AmbTempC = np.random.uniform(5.0,30.0) #25.0 # C
    geom.rock.AmbPres = 0.101 #Example: 0.01 MPa #Atmospheric: 0.101 # MPa
    geom.rock.ResE = np.random.uniform(60.0,70.0)*GPa #50.0*GPa
    geom.rock.Resv = np.random.uniform(0.20,0.30) #0.3
    geom.rock.Ks3 = np.random.uniform(0.6,0.7) #0.5
    geom.rock.Ks2 = geom.rock.Ks3 + np.random.uniform(0.1,0.25) # 0.75
    geom.rock.s3Azn = 0.1*deg
    geom.rock.s3AznVar = 0.1*deg
    geom.rock.s3Dip = 0.1*deg
    geom.rock.s3DipVar = 0.1*deg
    
    #fracture orientation parameters (wells will be NE and down from due North; North is min principal stress, s3)
    #   number of fractures in each set
    #       fNum[0] = Poles in NE & SW quadrant (i.e., sub perpendicular to wells)
    #       fNum[1] = Poles in NW & SE quadrant (i.e., sub parallel to wells)
    #       fNum[2] = Poles subvertical (i.e., sub-parallel to wells conjugate joint set)
    geom.rock.fNum = np.asarray([int(np.random.uniform(0,40)),
                            int(np.random.uniform(0,40)),
                            int(np.random.uniform(0,40))],dtype=int) #count
    #   permeable fractures tend to be > 50 m diameter, well spacing will be < 1000 m, fractures should be smaller (isolated) and larger (faults) than spacing
    geom.rock.fDia = np.asarray([[50.0,500.0],
                            [50.0,500.0],
                            [50.0,500.0]],dtype=float) #m
    #   see above for meaning of these orientations, note that dip is clockwise from strike #!!! this could be improved to be more statistically meaningful, but noncritical fractures wash out anyways
    spread = np.random.uniform(0,1,(3))*0.3
    walk = np.random.uniform(-1,1,(3))*0.3
    walk = walk*(1.0-spread)
    geom.rock.fStr = np.asarray([[(135.0+45.0*(walk[0]-spread[0]))*deg,(135.0+45.0*(walk[0]+spread[0]))*deg],
                            [(45.0+45.0*(walk[1]-spread[1]))*deg,(45.0+45.0*(walk[1]+spread[1]))*deg],
                            [(180.0+180.0*(walk[2]-spread[2]))*deg,(180.0+180.0*(walk[2]+spread[2]))*deg]],dtype=float) #m
    geom.rock.fDip = np.asarray([[(90.0+45.0*(walk[0]-spread[0]))*deg,(90.0+45.0*(walk[0]+spread[0]))*deg],
                            [(90.0+45.0*(walk[1]-spread[1]))*deg,(90.0+45.0*(walk[1]+spread[1]))*deg],
                            [(0.0+45.0*(walk[2]-spread[2]))*deg,(0.0+45.0*(walk[2]+spread[2]))*deg]],dtype=float) #m
    
    #fracture hydraulic aperture scaling and initialization parameters
    geom.rock.gamma = np.asarray([10.0**-3.0,10.0**-2.0,10.0**-1.2])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.000,0.200,0.800])
    geom.rock.b = np.asarray([0.999,1.0,1.001])
    geom.rock.N = np.asarray([0.0,0.6,2.0])
    geom.rock.alpha = np.asarray([2.0e-9,2.9e-8,10.0e-8])
    geom.rock.bh = np.asarray([0.00005,0.0005,0.002]) #np.asarray([0.00005,0.00010,0.00020])
    geom.rock.bh_min = 0.00005 #m
    geom.rock.bh_max = 0.01 #0.02000 #m
    geom.rock.bh_bound = np.random.uniform(0.001,0.005)
    geom.rock.f_roughness = np.random.uniform(0.7,1.0) #0.8
    
    #well parameters
    geom.rock.w_count = int(np.random.uniform(3,6)) #2 #production wells
    geom.rock.w_spacing = np.random.uniform(500.0,800.0) #300.0 #m
    geom.rock.w_length = np.random.uniform(800.0,1200.0) #800.0 #m
    geom.rock.w_azimuth = geom.rock.s3Azn + np.random.uniform(0.0,30.0)*deg #rad
    geom.rock.w_dip = geom.rock.s3Dip + np.random.uniform(0.0,30.0)*deg #rad
    geom.rock.w_proportion = np.random.uniform(0.8,0.4) #0.8 #m/m
    geom.rock.w_phase = np.random.uniform(0.0,360.0)*deg #-90.0*deg #rad
    geom.rock.w_toe = np.random.uniform(-15.0,15.0)*deg #rad
    geom.rock.w_skew = np.random.uniform(-15.0,15.0)*deg #rad
    geom.rock.w_intervals = int(np.random.uniform(4,9)) #3 #breaks in well length
    geom.rock.ra = 0.0254*np.random.uniform(4.0,8.0) #3.0 #0.0254*3.0 #m
    geom.rock.rgh = 80.0
    
    #cement properties
    geom.rock.CemKt = 2.0 # W/m-K
    geom.rock.CemSv = 2000.0 # kJ/m3-K
    
    #thermal-electric power parameters
    geom.rock.GenEfficiency = 0.85 # kWe/kWt
    geom.rock.LifeSpan = 20.5*yr #years
    geom.rock.TimeSteps = 41 #steps
    geom.rock.p_whp = 1.0*MPa #Pa
    geom.rock.Tinj = 95.0 #C
    geom.rock.H_ConvCoef = 3.0 #kW/m2-K
    geom.rock.dT0 = 10.0 #K
    geom.rock.dE0 = 500.0 #kJ/m2
    
    #water base parameters
    geom.rock.PoreRho = 980.0 #kg/m3 starting guess
    geom.rock.Poremu = 0.25*cP #Pa-s
    geom.rock.Porek = 0.1*mD #m2
    geom.rock.Frack = 100.0*mD #m2
    
    #stimulation parameters
    geom.rock.perf = 1
    geom.rock.r_perf = 50.0 #m
    geom.rock.sand = 0.3 #sand ratio in frac fluid
    geom.rock.leakoff = 0.0 #Carter leakoff
    geom.rock.dPp = -1.0*np.random.uniform(1.0,10.0)*MPa #-2.0*MPa #production well pressure difference from in-situ pore pressure
    geom.rock.dPi = 0.1*MPa #!!! #injection pressure increment for stimulation
    geom.rock.stim_limit = 5
#    geom.rock.Qinj = 0.01 #m3/s #this variable is changed in a unique way below since optimum flow rates are system specific
    geom.rock.Qstim = 0.00 #m3/s
    geom.rock.Vstim = 0.0 #m3
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
    u1 = np.random.uniform(20.0,55.0)*deg
    u2 = np.random.uniform(20.0,55.0)*deg
    u3 = 0.5*(u1+u2)
    geom.rock.phi = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])]) #rad
    # geom.rock.phi = np.asarray([25.0*deg,35.0*deg,45.0*deg]) #rad
    u1 = np.random.uniform(0.0,20.0)*MPa
    u2 = np.random.uniform(0.0,20.0)*MPa
    u3 = 0.5*(u1+u2)
    geom.rock.mcc = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])]) #Pa
    # geom.rock.mcc = np.asarray([5.0,10.0,15.0])*MPa #Pa
    geom.rock.hfmcc = np.random.uniform(0.0,0.2)*MPa #0.1*MPa
    geom.rock.hfphi = np.random.uniform(15.0,35.0)*deg #30.0*deg
    #**********************************

    #recalculate base parameters
    geom.rock.re_init()
    
    #generate domain
    geom.gen_domain()

    #generate natural fractures
    geom.gen_joint_sets()
    
    #generate wells
    wells = []
    geom.gen_wells(True,wells)
    
    # #stimulate
    # geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
    #               visuals=False,fname='stim')
    
    #test multiple randomly selected flow rates
    rates = np.random.uniform(0.02,0.045,1)
    #rates = [0.01]
    for r in rates:
        #copy base parameter set
        base = []
        base = copy.deepcopy(geom)
        
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
        x = base.save('inputs_results_random.txt',pin)


#show plots
pylab.show()