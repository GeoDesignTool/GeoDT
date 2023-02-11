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


#global parameter uncertainty randomizer & site specifications
for i in range(0,1):
    # ****************************************************************************
    #### model setup: UTAH FORGE --- DATA FROM MULTIPLE SOURCES
    # ****************************************************************************    
    #create model object
    geom = []
    geom = gt.mesh()
    
    #rock properties
    geom.rock.size = 1600.0 #m
    geom.rock.ResDepth = np.random.uniform(2340.0,2360.0)+1500.0 #6000.0 # m #!!!
    geom.rock.ResGradient = np.random.uniform(83.1,87.4) #50.0 #56.70 # C/km; average = 25 C/km
    geom.rock.ResRho = np.random.uniform(2550.0,2950.0) #2700.0 # kg/m3
    geom.rock.ResKt = np.random.uniform(2.27,3.58) #2.5 # W/m-K
    geom.rock.ResSv = np.random.uniform(0.7400,1.2000)*geom.rock.ResRho #2063.0 # kJ/m3-K
    geom.rock.AmbTempC = np.random.uniform(0.0,0.0) #25.0 # C
    geom.rock.AmbPres = 0.101 #Example: 0.01 MPa #Atmospheric: 0.101 # MPa
    geom.rock.ResE = np.random.uniform(55.0,62.0)*GPa #50.0*GPa
    geom.rock.Resv = np.random.uniform(0.26,0.40) #0.3 
    geom.rock.Ks3 = np.random.uniform(0.216,0.637) #0.5 
    geom.rock.Ks2 = np.max([np.random.uniform(0.250,0.750), geom.rock.Ks3+np.random.uniform(0.01,0.15)]) # 0.75 
    geom.rock.s3Azn = np.random.uniform(258.0,338.0)*deg 
    geom.rock.s3AznVar = 1.0*deg 
    geom.rock.s3Dip = np.random.uniform(-20.0,20.0)*deg 
    geom.rock.s3DipVar = 1.0*deg 
    #fracture orientation parameters #[i,:] set, [0,0:2] min, max --or-- nom, std
    geom.rock.fNum = np.asarray([int(np.random.uniform(0,35)),
                            int(np.random.uniform(0,60)),
                            int(np.random.uniform(0,15))],dtype=int) #count
    geom.rock.fDia = np.asarray([[150.0,1500.0],
                            [150.0,1500.0],
                            [150.0,1500.0]],dtype=float) #m
    geom.rock.fStr = np.asarray([[96.0*deg,8.0*deg],
                            [185.0*deg,8.0*deg],
                            [35.0*deg,8.0*deg,]],dtype=float) #m
    geom.rock.fDip = np.asarray([[80.0*deg,6.0*deg],
                            [48.0*deg,6.0*deg,],
                            [64.0*deg,6.0*deg]],dtype=float) #m
    #fracture hydraulic parameters #no data from FORGE available to populate fracture scaling parameters so universal values are applied 
    geom.rock.gamma = np.asarray([10.0**-3.0,10.0**-2.0,10.0**-1.2])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.000,0.200,0.800])
    geom.rock.b = np.asarray([0.999,1.0,1.001])
    geom.rock.N = np.asarray([0.0,0.6,2.0])
    geom.rock.alpha = np.asarray([2.0e-9,2.9e-8,10.0e-8])
    geom.rock.bh = np.asarray([0.00000001,0.00005,0.0001]) #np.asarray([0.00005,0.00010,0.00020])
    geom.rock.bh_min = 0.00000005 #m
    geom.rock.bh_max = 1.0 # 0.0001 #0.02000 #m
    geom.rock.bh_bound = np.random.uniform(0.0001,0.001)
    geom.rock.f_roughness = np.random.uniform(0.25**2,1.0) #0.8
    #well parameters
    geom.rock.w_count = 3 #production well #!!!
    geom.rock.w_length = 1113.9 #m 
    geom.rock.w_azimuth = 1.833 #rad 
    geom.rock.w_dip = 0.438 #rad 
    geom.rock.w_proportion = np.random.uniform(0.5,0.9) #m/m 
    geom.rock.w_phase = 1*90.0*deg #below #int(np.random.uniform(0.0,4.0))*90.0*deg #rad 
    geom.rock.w_toe = 0.0*deg #np.random.uniform(-5.0,5.0)*deg #rad 
    geom.rock.w_skew = 0.0*deg #np.random.uniform(-10.0,10.0)*deg #rad 
    geom.rock.w_intervals = 4 #int(np.random.uniform(1,7)) #!!!
    geom.rock.ra = 0.0889 #m #needs verification 
    geom.rock.rgh = 80.0 
    #cement properties
    geom.rock.CemKt = 2.0 # W/m-K 
    geom.rock.CemSv = 2000.0 # kJ/m3-K 
    #thermal-electric power parameters
    geom.rock.GenEfficiency = 1.00 # kWe/kWt 
    geom.rock.LifeSpan = 30.0*yr #years 
    geom.rock.TimeSteps = 40 #steps 
    geom.rock.p_whp = 1.0*MPa #Pa 
    geom.rock.Tinj = np.random.uniform(20.0,90.0) #C 
    geom.rock.H_ConvCoef = 3.0 #kW/m2-K 
    geom.rock.dT0 = 1.0 #K
    geom.rock.dE0 = 50.0 #kJ/m2
    #water base parameters
    geom.rock.PoreRho = np.random.uniform(920.0,932.0) #kg/m3 
    geom.rock.Poremu = 0.2*cP #Pa-s 
    geom.rock.Porek = 0.1*mD #m2 
    geom.rock.Frack = 100.0*mD #m2 
    #stimulation parameters
    geom.rock.perf = 1 
    geom.rock.sand = 0.3 #sand ratio in frac fluid 
    geom.rock.leakoff = 0.0 #Carter leakoff 
    geom.rock.dPp = np.random.uniform(2.0*MPa,-10.0*MPa) #production well pressure drawdown 
    geom.rock.dPi = 0.1*MPa 
    geom.rock.stim_limit = 5 #5 
    geom.rock.Qstim = geom.rock.Qinj #m3/s 
    geom.rock.Vstim = 400.0 #100000.0 #m3 
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
    geom.rock.phi = np.asarray([20.0*deg,35.0*deg,45.0*deg]) #rad 
    geom.rock.mcc = np.asarray([1.0,3.0,6.0])*MPa #Pa 
    geom.rock.hfmcc = np.random.uniform(0.1,0.4)*MPa #0.1*MPa 
    geom.rock.hfphi = np.random.uniform(15.0,35.0)*deg #30.0*deg 
    #*********************************
    #recalculate base parameters
    geom.rock.re_init()
    #generate domain
    geom.gen_domain()
    #generate natural fractures
    geom.gen_joint_sets()
    #copy site parameters with natural fractures populated
    site = []
    site = copy.deepcopy(geom)
            
    # ****************************************************************************
    #### varied design parameters
    # ****************************************************************************
    
    #investigate different well spacings - uniform sampling
    # spacings = list(np.random.uniform(30.0,747.8,5)) #!!!
    spacings = [550.0]
    for s in spacings:
        #get original site parameters
        geom = []
        geom = copy.deepcopy(site)
        #set well spacing and seed hydrofrac size r_perf
        geom.rock.w_spacing = s #m #will be varied below
        geom.rock.r_perf = 0.2*geom.rock.w_spacing #m 
        #generate wells
        wells = []
        geom.gen_wells(True,wells)        
        #copy geometry with placed wells
        base = []
        base = copy.deepcopy(geom)        
        
        #investigate different flow rates - logarithmic sampling
        # flows = list(10.0**(np.random.uniform(np.log10(0.0005),np.log10(0.1),5))) #!!!
        # flows = list(10.0**(np.random.uniform(np.log10(0.0005),np.log10(0.003),1)))
        flows = [0.015]
        for f in flows:
            #load geometry with wells placed
            geom = []
            geom = copy.deepcopy(base)
            pin = np.random.randint(100000000,999999999,1)[0]
            #solve for stimulation and circulation at specified circulation rate Qinj
            geom.rock.Qinj = f 
            geom.rock.re_init()
            try: #normal workflow if solution is successful
                geom.dyn_stim(Vinj=geom.rock.Vinj,Qinj=geom.rock.Qinj,target=[],
                                visuals=True,fname='run_%i' %(pin))
                geom.get_heat(plot=True,detail=True,lapse=True)
                plt.savefig('plt_%i.png' %(pin), format='png')
                plt.close()
                
                NPV, P, C, Q = geom.get_economics(detail=True)
                
                #save a 3D model of the scenario
                geom.build_vtk(fname='fin_%i' %(pin))
                
                #3D temperature visual (this is a slow process that can help with model validation, so it is not typically used)
                if False: 
                    geom.build_pts(spacing=100.0,fname='fin_%i' %(pin))
                
                #save primary inputs and outputs, if hydrofracs occur, they will be the last item in the list of fractures
                aux = [['type_last',geom.faces[-1].typ],
                        ['Pc_last',geom.faces[-1].Pc],
                        ['sn_last',geom.faces[-1].sn],
                        ['Pcen_last',geom.faces[-1].Pcen],
                        ['Pmax_last',geom.faces[-1].Pmax],
                        ['dia_last',geom.faces[-1].dia],
                        ['profit_sales',P],
                        ['cost_capital',C],
                        ['risk_quakes',Q],
                        ['NPV',geom.NPV]]
                geom.save('inputs_results_FORGE.txt',pin,aux=aux,printwells=0,time=True)
            
            except: #placeholder for failed models, bote that sometimes models fail for physical reasons (not just unhandled numerical errors)
                #(note that failed models can signifiy a failed feild test, so failures are a valid result)
                print( 'solver failure!')

#show plots
pylab.show()