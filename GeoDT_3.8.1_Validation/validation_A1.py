# ****************************************************************************
# validation: heat flow - two well - compare to Amanzi
# ****************************************************************************

# ****************************************************************************
#### standard imports
# ****************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import GeoDT as gt
import pylab
deg = gt.deg
MPa = gt.MPa
GPa = gt.GPa
yr = gt.yr
cP = gt.cP
mD = gt.mD

# ****************************************************************************
#### model setup
# ****************************************************************************
#create model object
geom = []
geom = gt.mesh()

#rock properties
geom.rock.size = 600.0 #m #!!!
geom.rock.ResDepth = 6000.0 # m
geom.rock.ResGradient = 50.0 # C/km; average = 25 C/km
geom.rock.ResRho = 2700.0 # kg/m3
geom.rock.ResKt = 2.5 # W/m-K
geom.rock.ResSv = 2063.0 # kJ/m3-K
geom.rock.AmbTempC = 25.0 # C
geom.rock.AmbPres = 0.101 #Example: 0.01 MPa #Atmospheric: 0.101 # MPa
geom.rock.ResE = 50.0*GPa
geom.rock.Resv = 0.3
geom.rock.Ks3 = 0.5
geom.rock.Ks2 = 0.75
geom.rock.s3Azn = 0.0*deg
geom.rock.s3AznVar = 0.0*deg
geom.rock.s3Dip = 0.0*deg
geom.rock.s3DipVar = 0.0*deg
geom.rock.fNum = np.asarray([int(0),
                        int(0),
                        int(0)],dtype=int) #count
geom.rock.fDia = np.asarray([[400.0,1200.0],
                        [200.0,1000.0],
                        [400.0,1200.0]],dtype=float) #m
#FORGE rotated 104 CCW
geom.rock.fStr = np.asarray([[351.0*deg,15.0*deg],
                        [80.0*deg,15.0*deg],
                        [290.0*deg,15.0*deg,]],dtype=float) #m
geom.rock.fDip = np.asarray([[80.0*deg,7.0*deg],
                        [48.0*deg,7.0*deg,],
                        [64.0*deg,7.0*deg]],dtype=float) #m
#fracture hydraulic parameters
geom.rock.gamma = np.asarray([10.0**-3.0,10.0**-2.0,10.0**-1.2])
geom.rock.n1 = np.asarray([1.0,1.0,1.0])
geom.rock.a = np.asarray([0.000,0.200,0.800])
geom.rock.b = np.asarray([0.999,1.0,1.001])
geom.rock.N = np.asarray([0.0,0.6,2.0])
geom.rock.alpha = np.asarray([2.0e-9,2.9e-8,10.0e-8])
geom.rock.bh = np.asarray([0.00005,0.0001,0.0002]) #np.asarray([0.00005,0.00010,0.00020])

geom.rock.bh_min = 0.0030001 #0.00005 #m #!!!
geom.rock.bh_max = 0.003 #0.01 #m #!!!
geom.rock.bh_bound = 0.001 #!!!
geom.rock.f_roughness = 0.8 #np.random.uniform(0.7,1.0) 
#well parameters
geom.rock.w_count = 1 #2 #wells
geom.rock.w_spacing = 300.0 #m
geom.rock.w_length = 100.0 #800.0 #m
geom.rock.w_azimuth = 0.0*deg #rad
geom.rock.w_dip = 0.0*deg #rad
geom.rock.w_proportion = 1.0 #m/m
geom.rock.w_phase = 0.0*deg #rad
geom.rock.w_toe = 0.0*deg #rad
geom.rock.w_skew = 0.0*deg #rad
geom.rock.w_intervals = 1 #breaks in well length
geom.rock.ra = 0.0254*5.0 #0.0254*3.0 #m
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
geom.rock.H_ConvCoef = 3.0 #kW/m2-K #!!!
geom.rock.dT0 = 10.0 #K
geom.rock.dE0 = 500.0 #kJ/m2
#water base parameters
geom.rock.PoreRho = 980.0 #kg/m3 starting guess #!!!
geom.rock.Poremu = 0.25*cP #Pa-s #!!!
geom.rock.Porek = 0.1*mD #m2
geom.rock.Frack = 100.0*mD #m2
#stimulation parameters
if geom.rock.w_intervals == 1:
    geom.rock.perf = int(np.random.uniform(1,1))
else:
    geom.rock.perf = 1
geom.rock.r_perf = 80.0 #m
geom.rock.sand = 0.3 #sand ratio in frac fluid
geom.rock.leakoff = 0.0 #Carter leakoff
geom.rock.dPp = -2.0*MPa #production well pressure drawdown #!!!
geom.rock.dPi = 0.1*MPa 
geom.rock.stim_limit = 5
geom.rock.Qinj = 0.003 #m3/s
geom.rock.Qstim = 0.01 #m3/s #0.08
geom.rock.Vstim = 150.0 #m3
geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
geom.rock.phi = np.asarray([35.0*deg,35.0*deg,35.0*deg]) #rad
geom.rock.mcc = np.asarray([10.0,10.0,10.0])*MPa #Pa
geom.rock.hfmcc = 0.0*MPa #0.1*MPa
geom.rock.hfphi = 30.0*deg #30.0*deg
geom.Kic = 1.5*MPa #Pa-m**0.5

# ****************************************************************************
#### model execution
# ****************************************************************************
#recalculate base parameters
geom.rock.re_init()

#generate domain
geom.gen_domain()

#generate natural fractures
#geom.gen_joint_sets()

#generate wells
wells = []
geom.gen_wells(True,wells)

#place fracture
c0 = [0.0,0.0,0.0]
dia = 800.0
strike = 90.0*deg
dip = 90.0*deg
geom.gen_fixfrac(True,c0,dia,strike,dip)
geom.fracs[-1].bh = 0.003

#setup internal variables for solver
geom.re_init()

#random identifier (statistically should be unique)
pin = np.random.randint(100000000,999999999,1)[0]

#stimulate
#geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
#              visuals=False,fname='stim')

#flow
geom.dyn_stim(Vinj=geom.rock.Vinj,Qinj=geom.rock.Qinj,target=[],
              visuals=False,fname='run_%i' %(pin))

#heat flow
geom.get_heat(plot=True)
plt.savefig('plt_%i.png' %(pin), format='png')
plt.close()

#show flow model
geom.build_vtk(fname='fin_%i' %(pin))
    
if True: #3D temperature visual
    geom.build_pts(spacing=20.0,fname='fin_%i' %(pin))
    
#save primary inputs and outputs
x = geom.save('inputs_results_valid.txt',pin)

#stereoplot
#geom.rock.stress.plot_Pc(geom.rock.phi[1], geom.rock.mcc[1], filename='Pc_stereoplot.png')

#check vs Detournay
#geom.detournay_visc(geom.rock.Qstim)

#show plots
pylab.show()