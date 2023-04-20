# ****************************************************************************
# validation: fracture cage experiments in acrylic
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
mLmin = 1.66667e-8 #m3/s

# ****************************************************************************
#### model setup
# ****************************************************************************
#create model object
geom = []
geom = gt.mesh()

#rock properties
geom.rock.size = 0.5*0.1524 #m
geom.rock.ResDepth = 0.0 # m
geom.rock.ResGradient = 25.0 # C/km; average = 25 C/km
geom.rock.ResRho = 1180.0 # kg/m3
geom.rock.ResKt = 0.5*(0.167+0.250) # W/m-K (mit.edu/~6.777/matprops/pmma.htm)
geom.rock.ResSv = 1730.0 # kJ/m3-K
geom.rock.AmbTempC = 19.0 # C
geom.rock.AmbPres = 0.078*MPa #Example: 0.01 MPa #Atmospheric: 0.101 # MPa
geom.rock.ResE = 2.55*GPa
geom.rock.Resv = 0.402
geom.rock.Ks3 = 0.5
geom.rock.Ks2 = 0.5
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
geom.rock.bh = np.asarray([0.00005,0.0001,0.0002])
geom.rock.bh_min = 0.0000010 #0.00005 #m
geom.rock.bh_max = 0.0040000 #0.01 #m
geom.rock.bh_bound = 0.005
geom.rock.f_roughness = 0.95 #0.8 #np.random.uniform(0.7,1.0) 
#well parameters
geom.rock.w_count = 3 #2 #wells
geom.rock.w_spacing = 0.040 #m
geom.rock.w_length = 0.0677 #800.0 #m
geom.rock.w_azimuth = 0.0*deg #rad
geom.rock.w_dip = 90.0*deg #rad
geom.rock.w_proportion = 0.0937 #m/m
geom.rock.w_phase = 0.0*deg #rad #CW from +x'?
geom.rock.w_toe = 0.0*deg #rad
geom.rock.w_skew = 0.0*deg #rad
geom.rock.w_intervals = 1 #breaks in well length
geom.rock.ra = 0.001524 #m
geom.rock.rb = 0.005000 #m
geom.rock.rc = 0.005556 #m
geom.rock.rgh = 80.0
#cement properties
geom.rock.CemKt = 0.2085 # W/m-K
geom.rock.CemSv = 1730.0 # kJ/m3-K
#thermal-electric power parameters
geom.rock.GenEfficiency = 0.85 # kWe/kWt
geom.rock.LifeSpan = 0.1*yr #years
geom.rock.TimeSteps = 41 #steps
geom.rock.p_whp = 0.25*MPa #Pa
geom.rock.Tinj = 50.0 #C
geom.rock.H_ConvCoef = 3.0 #kW/m2-K
geom.rock.dT0 = 2.0 #K
geom.rock.dE0 = 500.0 #kJ/m2
#water base parameters
geom.rock.PoreRho = 880.0 #kg/m3
geom.rock.Poremu = 404*cP #Pa-s
geom.rock.Porek = 0.1*mD #m2
geom.rock.Frack = 100.0*mD #m2
#stimulation parameters
if geom.rock.w_intervals == 1:
    geom.rock.perf = int(np.random.uniform(1,1))
else:
    geom.rock.perf = 1
geom.rock.r_perf = 0.007 #m
geom.rock.sand = 0.0 #sand ratio in frac fluid
geom.rock.leakoff = 0.0 #Carter leakoff
geom.rock.dPp = 0.25*MPa #production well pressure drawdown
geom.rock.dPi = 0.1*MPa #0.75*MPa #0.1*MPa #!!!  First successful at 0.5 mL/min using dPi=0.5*MPa and stim_limit=5
geom.rock.stim_limit = 5 #5
geom.rock.Qinj = 8.0*mLmin #m3/s
geom.rock.Qstim = 0.5*mLmin #m3/s
geom.rock.Vstim = geom.rock.Qstim*30.0*60.0 #m3
geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
geom.rock.phi = np.asarray([30.0*deg,30.0*deg,30.0*deg]) #rad
geom.rock.mcc = np.asarray([2.0,2.0,2.0])*MPa #Pa
geom.rock.hfmcc = 2.00*MPa #0.1*MPa
geom.rock.hfphi = 30.0*deg #30.0*deg
geom.Kic = 2.2*MPa #Pa-m**0.5

# ****************************************************************************
#### model execution
# ****************************************************************************
#create model backup
base = []
base = copy.deepcopy(geom)

#tighten wells
#geom.rock.ra = 0.75*0.001524 #m

#variants
num_wells = [1,2,3,4]
AznDip = [[15.0,0.0],[75.0,-20.0],[0.0,90.0]] #vertical, sub-vertical, horizontal hydrofracs
flows = [0.5*mLmin,8.0*mLmin] #low flow vs high flow
ras = [geom.rock.ra,0.4*geom.rock.ra,0.3*geom.rock.ra,0.2*geom.rock.ra,0.1*geom.rock.ra] #standard vs choked

#well layouts
bit = 0
for w in range(0,len(num_wells)):
    bit += 1 

    #hydrofrac orientations
    wit = 0
    for a in range(0,len(AznDip)):
        
        #flow rates
        for f in range(0,len(flows)):
            
            #well inner diameters
            for r in range(0,len(ras)):
                wit += 1
                #fetch base model
                geom = []
                geom = copy.deepcopy(base)
                
                #set well layout
                geom.rock.w_count = num_wells[w]
                
                #set hydrofrac angle
                geom.rock.s3Azn = AznDip[a][0]*deg
                geom.rock.s3Dip = AznDip[a][1]*deg
                
                #set flow rate
                geom.rock.Qinj = flows[f]
                
                #identifier
                pin = int('%i%i%i%i' %(w,a,f,r))
               
                #recalculate base parameters
                geom.rock.re_init()
                
                #generate domain
                geom.gen_domain()
                
                #generate natural fractures
                geom.gen_joint_sets()
                
                #generate wells
                wells = []
                geom.gen_wells(True,wells)
                
                #override well dimensions
                for i in geom.wells:
                    i.ra = ras[r]
                    i.rb = geom.rock.rb
                    i.rc = geom.rock.rc
                geom.wells[0].ra = geom.rock.ra
                
            
                #setup internal variables for solver
                geom.re_init()
                
                # #stimulate
                # geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
                #               visuals=False,fname='stim_w%i_s%i' %(w,a))
                
                #circulate
                geom.dyn_stim(Vinj=geom.rock.Vinj,Qinj=geom.rock.Qinj,target=[],
                              visuals=False,fname='circ_%04d' %(pin))
        
                #heat flow
                geom.get_heat(plot=True)
                plt.savefig('plot_%04d.png' %(pin), format='png')
                plt.close()
            
                #show flow model
                if bit+wit == 2:
                    geom.build_vtk(fname='fini_%04d' %(pin),vtype=[0,0,0,0,0,1])
                if wit == 1:
                    geom.build_vtk(fname='fini_%04d' %(pin),vtype=[1,0,0,0,0,0])
                geom.build_vtk(fname='fini_%04d' %(pin),vtype=[0,0,1,1,1,0])
                
                # #save primary inputs and outputs
                # aux = []
                # aux += [['qi_2',geom.q[2]]]
                # aux += [['qi_5',geom.q[5]]]
                # aux += [['qi_6',geom.q[6]]]
                #x = geom.save('inputs_results_valid.txt',pin,aux=aux,printwells=4)
                aux = [['type_last',geom.faces[-1].typ],
                       ['Pc_last',geom.faces[-1].Pc],
                       ['sn_last',geom.faces[-1].sn],
                       ['Pcen_last',geom.faces[-1].Pcen],
                       ['Pmax_last',geom.faces[-1].Pmax],
                       ['dia_last',geom.faces[-1].dia]]
                geom.save('inputs_results_valid.txt',pin,aux=aux,printwells=0)

if False: #3D temperature visual
    geom.build_pts(spacing=0.005,fname='dist_%04d' %(pin))

#stereoplot
#geom.rock.stress.plot_Pc(geom.rock.phi[1], geom.rock.mcc[1], filename='Pc_stereoplot.png')

#check vs Detournay
#geom.detournay_visc(geom.rock.Qstim)

#show plots
pylab.show()