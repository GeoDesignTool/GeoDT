# ****************************************************************************
#### GLADE AGS and EGS Design Evaluation - Luke Frash // Los Alamos
# ****************************************************************************

# ****************************************************************************
#### standard imports
# ****************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import GeoDT as gt
import pylab
import copy

#units (standard will be: N, m, kg, s, K, Pa, etc.)
deg = gt.deg
MPa = gt.MPa
GPa = gt.GPa
yr = gt.yr
cP = gt.cP
mD = gt.mD
mLmin = gt.mLmin
um2cm = gt.um2cm
gal = 1.0/264.172 #m3

#global parameter uncertainty randomizer & site specifications
for i in range(0,1):
    # ****************************************************************************
    #### model setup: UTAH FORGE --- DATA FROM MULTIPLE SOURCES
    # ****************************************************************************
    #generate pin
    pin = np.random.randint(100000000,999999999,1)[0]
    
    #create model object
    geom = []
    geom = gt.mesh()
    
    #GLADE design parameters to modify per batch run
    geom.rock.w_count = 3 #production well #!!!
    geom.rock.w_phase = 1*90.0*deg #3pi = above, 1pi = below #rad #!!!
    geom.rock.perf_clusters = 3 #number of peforation clusters #!!!
    geom.rock.sand = 0.000 #0.044 #sand ratio in frac fluid by volume #!!!
    geom.rock.w_intervals = 6 #int(np.random.uniform(1,7)) #!!!
    
    #rock properties
    geom.rock.size = 2500.0 #m                                                  #half-size of the modeling domain
    geom.rock.gradient = True                                                   #use gethermal gradient in reservoir
    geom.rock.ResDepth = 6100 #5500.0  #m  3400 to 6100                         #depth of the reservoir
    geom.rock.ResGradient = np.random.uniform(36.0,50.0) #C/km                  #geothermal gradient
    geom.rock.ResRho = np.random.uniform(2550.0,2950.0) #kg/m3                  #rock density
    geom.rock.ResKt = np.random.uniform(2.27,3.58) #W/m-K                       #rock thermal conductivity
    geom.rock.ResSv = np.random.uniform(0.74,1.20)*geom.rock.ResRho ##kJ/m3-K   #rock specific heat capacity
    geom.rock.AmbTempC = np.random.uniform(0.0,0.0) #C                          #surface ambient air temperature
    geom.rock.AmbPres = 0.101*MPa #Example: 0.01 MPa #Atmospheric: 0.101 # MPa  #wanna know more? look at the docs!
    geom.rock.ResE = np.random.uniform(55.0,62.0)*GPa #50.0*GPa
    geom.rock.Resv = np.random.uniform(0.26,0.40) #
    geom.rock.Ks3 = np.random.uniform(0.300,0.800)
    geom.rock.Ks2 = np.max([np.random.uniform(0.400,1.100), geom.rock.Ks3+np.random.uniform(0.10,0.20)])
    geom.rock.s3Azn = np.random.uniform(60.0,90.0)*deg #based on world stress map - Ft Collins breakout data
    geom.rock.s3AznVar = 0.75*deg 
    geom.rock.s3Dip = np.random.uniform(0.0,30.0)*deg  #np.random.uniform(-20.0,20.0)*deg 
    geom.rock.s3DipVar = 0.75*deg 
    #fracture density and size (orientations to be determined by stress state)
    geom.rock.fNum = np.asarray([int(np.random.uniform(40,80)),
                            int(np.random.uniform(20,40)),
                            int(np.random.uniform(10,20))],dtype=int) #count
    geom.rock.fNum = np.asarray([int(np.random.uniform(0,0)),
                            int(np.random.uniform(0,0)),
                            int(np.random.uniform(0,0))],dtype=int) #count
    geom.rock.fDia = np.asarray([[300.0,3000.0],
                            [300.0,3000.0],
                            [300.0,3000.0]],dtype=float) #m
    #fracture hydraulic parameters with universal values
    geom.rock.gamma = np.asarray([10.0**-3.0,10.0**-2.0,10.0**-1.2])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.000,0.200,0.800])
    geom.rock.a = np.asarray([0.000,0.0025,0.0050])
    geom.rock.b = np.asarray([0.999,1.0,1.001])
    geom.rock.N = np.asarray([0.0,0.6,2.0])
    geom.rock.alpha = np.asarray([2.0e-9,2.9e-8,10.0e-8])
    geom.rock.prop_alpha = np.asarray([2.0e-9,2.9e-8,10.0e-8])
    geom.rock.bh = np.asarray([0.00000001,0.00005,0.0001]) 
    geom.rock.bh_min = 0.00000005 #m
    geom.rock.bh_max = 1.0 # 0.0001 #0.02000 #m
    geom.rock.bh_bound = np.random.uniform(0.0001,0.001)
    geom.rock.f_roughness = [0.25**2,0.5**2,1.0] #np.random.uniform(0.25**2,1.0) #0.8
    #well parameters
    geom.rock.w_azimuth = 75.0*deg #rad 
    geom.rock.w_dip = 15.0*deg #45.0*deg #rad 
    geom.rock.w_proportion = 0.6 #m/m 
    geom.rock.w_length = (1200.0/np.sin(geom.rock.w_dip))/geom.rock.w_proportion #m 
    geom.rock.w_length = 1600.0/geom.rock.w_proportion #m
    geom.rock.w_toe = 0.0*deg #np.random.uniform(-5.0,5.0)*deg #rad 
    geom.rock.w_skew = 0.0*deg #np.random.uniform(-10.0,10.0)*deg #rad 
    # geom.rock.ra = 0.06985 #m 
    # geom.rock.rb = 0.08255 #m 
    # geom.rock.rc = 0.10795 #m 8.5 in bore
    geom.rock.ra = 0.1397 #m 
    geom.rock.rb = 0.1524 #m 
    geom.rock.rc = 0.1778 #m 14 in bore 
    geom.rock.rgh = 80.0 
    #cement properties
    geom.rock.CemKt = 2.0 # W/m-K 
    geom.rock.CemSv = 2000.0 # kJ/m3-K 
    #thermal-electric power parameters
    geom.rock.GenEfficiency = 0.85 # kWe/kWt
    geom.rock.LifeSpan = 15.0*yr #years #typical EGS service life should be 10-40 years
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
    #stimulation parameters
    geom.rock.kf = 10.0*um2cm #m2 #100 um2cm = 0.11 mD
    geom.rock.perf_dia = 0.015 #np.random.uniform(0.010,0.020) #0.019 #m #number of peforation clusters
    geom.rock.perf_per_cluster = 3 #m #number of peforation clusters
    geom.rock.leakoff = 0.0 #Carter leakoff 
    geom.rock.dPp = -2.0*MPa #np.random.uniform(0.0*MPa,-3.0*MPa) #production well pressure drawdown 
    geom.rock.dPi = 0.02*MPa #0.1*MPa 
    geom.rock.stim_limit = 1 #5 #5 
    geom.rock.Qstim = geom.rock.Qinj #m3/s 
    geom.rock.Vstim = 400.0 #100000.0 #m3
    geom.rock.pfinal_max = 999.9*MPa #Pa
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
    geom.rock.phi = np.asarray([20.0*deg,35.0*deg,45.0*deg]) #rad 
    geom.rock.mcc = np.asarray([1.0,3.0,6.0])*MPa #Pa 
    geom.rock.hfmcc = np.random.uniform(0.1,0.4)*MPa #0.1*MPa 
    geom.rock.hfphi = np.random.uniform(15.0,35.0)*deg #30.0*deg 
    #recalculate base parameters
    geom.rock.re_init()
    
    # stress stereoplots
    # geom.rock.stress.plot_Pc(geom.rock.phi, geom.rock.mcc, filename='Pc_stereoplot.png')
    
    # ****************************************************************************
    #### model initializtion stuff
    # ****************************************************************************
    
    #impose maximum injection pressure boundary condition (if not hydropropping)
    if False: #if long term injection is not to be allowed above the hydraulic fracture gradient, make this "True"
        geom.rock.pfinal_max = 0.9*geom.rock.s3 #!!!
    #generate domain
    geom.gen_domain()
    #generate natural fractures
    geom.gen_joint_sets()
    #copy site parameters with natural fractures populated
    site = []
    site = copy.deepcopy(geom)
    #print the fracture geometry
    geom.re_init()
    geom.build_vtk(fname='start_%i' %(pin),vtype=[0,1,0,0,0,0]) #show natural fractures
    if (i == 0):
        geom.build_vtk(fname='start_%i' %(pin),vtype=[0,0,0,0,0,1]) #show model boundary
            
    # ****************************************************************************
    #### varied design parameters
    # ****************************************************************************
    
    #investigate different well spacings - uniform sampling
    spacings = list(np.random.uniform(400.0,600.0,1)) #!!!
    for s in spacings:
        #get original site parameters
        geom = []
        geom = copy.deepcopy(site)
        #set well spacing and seed hydrofrac size r_perf
        geom.rock.w_spacing = s #m #will be varied below
        geom.rock.r_perf = 0.2*geom.rock.w_spacing #m 
        
        #***** generate wells ****** #!!!
        wells = []
        # geom.gen_wells(True,wells,style='AGS') #AGS designs
        # geom.gen_wells(True,wells,style='EGS') #EGS designs
        geom.gen_wells(True,wells,style='IGS') #Isolated injectors
               
        #copy geometry with placed wells
        base = []
        base = copy.deepcopy(geom)      
        
        #investigate different flow rates - logarithmic sampling
        flows = list((10.0**(np.random.uniform(np.log10(0.10),np.log10(0.12),1)))/geom.rock.w_intervals) #!!!
        for f in flows:
            #load geometry with wells placed
            geom = []
            geom = copy.deepcopy(base)
            #solve for stimulation and circulation at specified circulation rate Qinj
            geom.rock.Qinj = f 
            geom.rock.re_init()
            if True:
            # try: #normal workflow if solution is successful
                geom.dyn_stim(Vinj=geom.rock.Vinj,Qinj=geom.rock.Qinj,target=[],
                                visuals=False,fname='run_%i' %(pin))
                geom.get_heat(plot=True,detail=True,lapse=False)
                plt.savefig('plt_%i.png' %(pin), format='png')
                plt.close()
                
                NPV, P, C, Q = geom.get_economics(detail=True) #sales must be removed to estimate $/MWh
                
                #save primary inputs and outputs
                aux = []
                geom.save('inputs_results.txt',pin,aux=aux,printwells=0,time=True)
            else:
            # except: #placeholder for failed models, bote that sometimes models fail for physical reasons (not just unhandled numerical errors)
                #(note that failed models can signifiy a failed field test, so failures are a valid result)
                print( 'solver failure!')
            #print first
            if (f == flows[0]) and (s == spacings[0]):
                #geometry only for first run
                geom.build_vtk(fname='fin_%i' %(pin),vtype=[1,0,1,1,0,0]) #well, flowing fractures, nodes
            #pipes for all
            geom.build_vtk(fname='fin_%i' %(pin),vtype=[0,0,0,0,1,0]) #pipes
            #new pin
            pin = np.random.randint(100000000,999999999,1)[0]

#show plots
pylab.show()