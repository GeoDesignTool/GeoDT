# -*- coding: utf-8 -*-

# ****************************************************************************
# Calculate economic potential of EGS & optimize borehole layout with caging
# Author: Luke P. Frash
#
# Notation:
#  @@@ for code needing to be updated
#  *** for breaks betweeen key sections
# ****************************************************************************

# ****************************************************************************
#### libraries
# ****************************************************************************
import numpy as np
from scipy.linalg import solve
#from scipy.stats import lognorm
import pylab
import math
from iapws import IAPWS97 as therm
import SimpleGeometry as sg
from scipy import stats
#import sys
import matplotlib.pyplot as plt
import copy

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
mD=darcy*10.0**-3.0#m2
yr=365.2425*24.0*60.0*60.0#s
pi=math.pi

# ****************************************************************************
#### classes, functions, and modules
# ****************************************************************************
# Place a radial hydraulic fracture of radius r at x0
def HF(r,x0, strikeRad, dipRad, h=0.5):
    # start with a disk
    disk=sg.diskObj(r,h)
    disk=sg.rotateObj(disk,[0.0,1.0,0.0],dipRad)
    disk=sg.rotateObj(disk,[0.0,0.0,1.0],-strikeRad)
    disk=sg.transObj(disk,x0)
    return disk

#definitions and cross referencing for pipe types
def typ(key):
    ret = []
    choices = np.asarray([
            ['boundary', '-3'],
            ['producer', '-2'],
            ['injector', '-1'],
            ['pipe', '0'],
            ['fracture', '1'],
            ['propped', '2'],
            ['darcy', '3'],
            ['choke', '4']
            ])
    key = str(key)
    ret = np.where(choices == key)
    if ret[1] == 0:
        ret = int(choices[ret[0],1][0])
    elif ret[1] == 1:
        ret = str(choices[ret[0],0][0])
    else:
        print('**invalid pipe type defined**')
        ret = []
    return ret

#azn and dip from endpoints
def azn_dip(x0,x1):
    dx = x1[0] - x0[0]
    dy = x1[1] - x0[1]
    dz = x1[2] - x0[2]
    dr = (dx**2.0+dy**2.0)**0.5
    azn = []
    dip = []
    if dy == 0:
        azn = np.sign(dx)*pi/2.0
    else:
        azn = np.arctan(dx/dy)
    if dr == 0.0:
        dip = np.sign(dz)*pi/2.0
    else:
        dip = np.arctan(dz/dr)
    return azn, dip

class cauchy: #functions modified from JPM
    def __init__(self):
        self.sigP = np.zeros((3,3),dtype=float)
        self.sigG = np.zeros((3,3),dtype=float)
        self.Sh = 0.0 #Pa
        self.SH = 0.0 #Pa
        self.SV = 0.0 #Pa
        self.Sh_azn = 0.0*deg #rad
        self.Sh_dip = 0.0*deg #rad
    # http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    def rotationMatrix(self, axis, theta):
        #return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
        axis = np.asarray(axis)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    #http://www.continuummechanics.org/stressxforms.html
    def rotateTensor(self, tensor, axis, theta):
        rot=self.rotationMatrix(axis,theta)
        return np.dot(rot,np.dot(tensor,np.transpose(rot)))
    #Projection of normal plane from dip directions
    def normal_from_dip(self, dip_direction,dip_angle):
            # dip_direction=0 is north -> nrmxH=0, nrmyH=1
            # dip_direction=90 is east -> nrmxH=1, nrmyH=0
            nrmxH=np.sin(dip_direction)
            nrmyH=np.cos(dip_direction)
            # The lateral components of the normals are corrected for the dip angle
            nrmx=nrmxH*np.sin(dip_angle)
            nrmy=nrmyH*np.sin(dip_angle)
            # The vertical
            nrmz=np.cos(dip_angle)
            return np.asarray([nrmx,nrmy,nrmz])
    #normal stress and critical slip or opening pressure
    def Pc(self, nrmP, phi, mcc):
        # Shear traction on the fault segment
        t=np.zeros([3])
        for i in range(3):
            for j in range(3):
                t[i] += self.sigG[j][i]*nrmP[j]
        # Normal component of the traction
        Sn=0.0;
        for i in range(3):
            Sn += t[i]*nrmP[i]
        # Shear component of the traction
        tauV=np.zeros([3])
        for i in range(3):
            tauV[i]=t[i]-Sn*nrmP[i]
        tau=np.sqrt(tauV[0]*tauV[0]+tauV[1]*tauV[1]+tauV[2]*tauV[2])
        # Critical pressure for slip from mohr-coulomb
        Pc1 = Sn - (tau-mcc)/np.tan(phi)
        # Critical pressure for tensile opening
        Pc2 = Sn + mcc
        # Critical pressure for fracture activation
        Pc = np.min([Pc1,Pc2])
        return Pc, Sn, tau
    #critical slip given fracture strike and dip
    def Pc_frac(self, strike, dip, phi, mcc):
        #get fractrue normal vector
        nrmG=self.normal_from_dip(strike+np.pi/2, dip)
        return self.Pc(nrmG, phi, mcc)
    #Set cauchy stress tensor from rotated principal stresses
    def set_sigG_from_Principal(self,Sh,SH,SV,ShAznDeg,ShDipDeg):
        # Stresses in the principal stress directions
        # We have been given the azimuth of the minimum stress, ShAzimuthDeg, to compare with 90 (x-dir)
        # That is Sh==Sxx, SH=Syy, SV=Szz in the principal stress coord system
        sigP=np.asarray([
            [Sh,  0.0, 0.0],
            [0.0, SH,  0.0],
            [0.0, 0.0, SV ]
            ])
        # Rotate about z-axis
        deltaShAz=(ShAznDeg-90.0)*np.pi/180.0
        # Rotate about y-axis
        ShDip=-ShDipDeg*np.pi/180.0
        sigG=self.rotateTensor(sigP,[0.0,1.0,0.0],-ShDip)
        sigG=self.rotateTensor(sigG,[0.0,0.0,1.0],-deltaShAz)
        self.sigP = sigP
        self.sigG = sigG
        self.Sh = Sh #Pa
        self.SH = SH #Pa
        self.SV = SV #Pa
        self.Sh_azn = ShAznDeg*deg #rad
        self.Sh_dip = ShDipDeg*deg #rad
        return sigG
    #Plot citical pressure
    def plot_Pc(self, phi, mcc, filename='Pc_stereoplot.png'):
        # Working variables
        nRad=100
        dip_angle_deg = np.asarray(range(nRad+1))*(90.0/nRad)
        nTheta=200
        dip_dir_radians = np.asarray(range(nTheta+1))*(2.0*np.pi/nTheta)
        png_dpi=128
        # Calculate critical dP
        criticalDelPpG=np.zeros([nRad,nTheta])
        for i in range(nRad):
            for j in range(nTheta):
                # Convert the x and y into a normal to the fracture using dip angle and dip direction
                nrmG=self.normal_from_dip(dip_dir_radians[j], dip_angle_deg[i]*deg)
                nrmx,nrmy,nrmz=nrmG[0],nrmG[1],nrmG[2]
                criticalDelPpG[i,j], h1, h2 = self.Pc(np.asarray([nrmx,nrmy,nrmz]), phi, mcc)
        # Plot critical slip pressure (lower hemisphere projection)
        fig = pylab.figure(figsize=(6,4.75),dpi=png_dpi,tight_layout=True,facecolor='w',edgecolor='k')
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        pylab.pcolormesh(dip_dir_radians+np.pi,dip_angle_deg,np.ma.masked_where(np.isnan(criticalDelPpG),criticalDelPpG), vmin=110.0*MPa, vmax=150.0*MPa,cmap='rainbow_r') #vmin=0.0*MPa, vmax=self.SH,cmap='rainbow_r')
        ax.grid(True)
        ax.set_rgrids([0,30,60,90],labels=[])
        ax.set_thetagrids([0,90,180,270])
        pylab.colorbar()
        ax.set_title("Critical pressure for Mohr-Coulomb failure", va='bottom')
        pylab.savefig(filename, format='png',dpi=png_dpi)
        pylab.close()
        
#reservoir object
class reservoir:
    def __init__(self):
        #rock properties
        self.size = 800.0 #m
        self.ResDepth = 6000.0 # m
        self.ResGradient = 50.0 #56.70 # C/km; average = 25 C/km
        self.ResRho = 2700.0 # kg/m3
        self.ResKt = 2.5 # W/m-K
        self.ResSv = 2063.0 # kJ/m3-K
        self.AmbTempC = 25.0 # C
        self.AmbPres = 0.101 #Example: 0.01 MPa #Atmospheric: 0.101 # MPa
        self.ResE = 50.0*GPa
        self.Resv = 0.3
        self.ResG = self.ResE/(2.0*(1.0+self.Resv))
        self.Ks3 = 0.5
        self.Ks2 = 0.75 # 0.75
        self.s3Azn = 0.0*deg
        self.s3AznVar = 0.0*deg
        self.s3Dip = 0.0*deg
        self.s3DipVar = 0.0*deg
        #fracture orientation parameters #[i,:] set, [0,0:2] min, max --or-- nom, std
        self.fNum = np.asarray([10,
                                10,
                                10],dtype=int) #count 
        self.fDia = np.asarray([[300.0,900.0],
                                [300.0,900.0],
                                [300.0,900.0]],dtype=float) #m
        self.fStr = np.asarray([[79.0*deg,8.0*deg],
                                [0.0*deg,8.0*deg],
                                [180.0*deg,8.0*deg]],dtype=float) #m
        self.fDip = np.asarray([[60.0*deg,8.0*deg],
                                [90.0*deg,8.0*deg],
                                [0.0*deg,8.0*deg]],dtype=float) #m
        #fracture hydraulic parameters
        self.alpha = np.asarray([-0.002/MPa,-0.028/MPa,-0.080/MPa])
        self.gamma = np.asarray([0.005,0.01,0.05])
        self.n1 = np.asarray([1.0,1.0,1.0])
        self.a = np.asarray([0.031,0.05,0.125])
        self.b = np.asarray([0.75,0.8,0.85])
        self.N = np.asarray([0.2,0.5,1.2])
        self.bh = np.asarray([0.00005,0.0001,0.0002]) #np.asarray([0.00005,0.00010,0.00020])
        self.bh_min = 0.00005 #m
        self.bh_max = 0.02 #0.02000 #m
        self.bh_bound = 0.003
        self.f_roughness = 0.8
        #well parameters
        self.w_count = 3 #wells
        self.w_spacing = 200.0 #m
        self.w_length = 800.0 #m
        self.w_azimuth = self.s3Azn + 0.0*deg #rad
        self.w_dip = self.s3Dip + 0.0*deg #rad
        self.w_proportion = 0.8 #m/m
        self.w_phase = -45.0*deg #rad
        self.w_toe = 0.0*deg #rad
        self.w_skew = 15.0*deg #rad
        self.w_intervals = 5 #breaks in well length
        self.ra = 0.0254*3.0 #0.0254*3.0 #m
        self.rb = self.ra + 0.0254*0.5 # m
        self.rc = self.ra + 0.0254*1.0 # m
        self.rgh = 80.0
        #cement properties
        self.CemKt = 2.0 # W/m-K
        self.CemSv = 2000.0 # kJ/m3-K
        #thermal-electric power parameters
        self.GenEfficiency = 0.85 # kWe/kWt
#        self.InjPres = 1.0 #Example: 0.135 #Model: 2.0 # MPa
#        self.TargetPower = 1000 #Example: 2964 # kWe
        self.LifeSpan = 20.5*yr #years
        self.TimeSteps = 41 #steps
        self.p_whp = 1.0*MPa #Pa
        self.Tinj = 95.0 #C
        self.H_ConvCoef = 3.0 #kW/m2-K
        self.dT0 = 10.0 #K
        self.dE0 = 500.0 #kJ/m2
        #water base parameters
        self.PoreRho = 980.0 #kg/m3 starting guess
        self.Poremu = 0.9*cP #Pa-s
        self.Porek = 0.01*mD #m2
        self.Frack = 100.0*mD #m2
        #calculated parameters
        self.BH_T = self.ResDepth*10**-3.0*self.ResGradient + self.AmbTempC + 273.15 #K
        self.BH_P = self.PoreRho*g*self.ResDepth #Pa
        self.s1 = self.ResRho*g*self.ResDepth #Pa
        self.s2 = self.Ks2*(self.s1-self.BH_P)+self.BH_P #Pa
        self.s3 = self.Ks3*(self.s1-self.BH_P)+self.BH_P #Pa
        #cauchy stress
        self.stress = cauchy()
        self.stress.set_sigG_from_Principal(self.s3, self.s2, self.s1, self.s3Azn, self.s3Dip)
#        self.stress.plot_Pc(30.0*deg,5.0*MPa)
        #stimulation parameters
        self.perf = 1
        self.r_perf = 50.0 #m
        self.sand = 0.3 #sand ratio in frac fluid
        self.leakoff = 0.0 #Carter leakoff
        self.dPp = -2.0*MPa #production well pressure drawdown
        self.dPi = 0.5*MPa 
        self.stim_limit = 5
        self.Qinj = 0.01 #m3/s
        self.Vinj = self.Qinj*self.LifeSpan
        self.Qstim = 0.04 #m3/s
        self.Vstim = 50000.0 #m3
        self.bval = 1.0 #Gutenberg-Richter magnitude scaling
        self.phi = np.asarray([20.0*deg,35.0*deg,50.0*deg]) #rad
        self.mcc = np.asarray([5.0*MPa,10.0*MPa,15.0*MPa]) #Pa
        self.hfmcc = 0.1*MPa
        self.hfphi = 30.0*deg
        self.Kic = 1.5*MPa #Pa-m**0.5
        #matrix-flow parameters
        self.m_dP0 = 0.10*MPa #Pa
        self.m_porosity = 0.05 #unitless
        self.m_Kr = self.ResE/(3.0*(1.0-2.0*self.Resv)) #Pa
        self.m_Kf = 2.15*GPa #Pa
        self.m_Ct = self.m_porosity/self.m_Kf + (1.0 - self.m_porosity)/self.m_Kr
        
    def re_init(self):
        #calculated parameters
        self.ResG = self.ResE/(2.0*(1.0+self.Resv))
        self.BH_T = self.ResDepth*10**-3.0*self.ResGradient + self.AmbTempC + 273.15 #K
        self.BH_P = self.PoreRho*g*self.ResDepth #Pa
        self.s1 = self.ResRho*g*self.ResDepth #Pa
        self.s2 = self.Ks2*(self.s1-self.BH_P)+self.BH_P #Pa
        self.s3 = self.Ks3*(self.s1-self.BH_P)+self.BH_P #Pa
        self.stress.set_sigG_from_Principal(self.s3, self.s2, self.s1, self.s3Azn, self.s3Dip)
        self.Vinj = self.Qinj*self.LifeSpan
        self.rb = self.ra + 0.0254*0.5 # m
        self.rc = self.ra + 0.0254*1.0 # m
        self.m_Kr = self.ResE/(3.0*(1.0-2.0*self.Resv)) #Pa
        self.m_Ct = self.m_porosity/self.m_Kf + (1.0 - self.m_porosity)/self.m_Kr
        
#surface object
class surf:
    def __init__(self,x0=0.0, y0=0.0, z0=0.0, dia=1.0, stk=0.0*deg, dip=90.0*deg,
                 ty='fracture', rock = reservoir(),
                 mcc = -1, phi = -1):
        #*** base parameters ***
        #node number of center point
        self.ci = -1
        #geometry
        self.c0 = np.asarray([x0, y0, z0])
        self.dia = dia
        self.str = stk
        self.dip = dip
        self.typ = typ(ty)
        #shear strength
        self.phi = -1.0 #rad
        self.mcc = -1.0 # Pa
        #stress state
        self.sn = 5.0*MPa
        self.En = 50.0*GPa
        self.vn = 0.30
        self.Pc = 0.0*MPa
        self.tau = 0.0*MPa
        #stimulation information
        self.stim = 0
        self.Pmax = 0.0*MPa
        self.Pcen = 0.0*MPa
        self.Mws = []
        self.hydroprop = False
        #scaling
        self.u_N = -1.0
        self.u_alpha = -1.0
        self.u_a = -1.0
        self.u_b = -1.0
        self.u_gamma = -1.0
        self.u_n1 = -1.0
        #hydraulic geometry
        self.bh = -1.0
        #matrix leakoff
        self.m_leakvol0 = 0.0 #m3
        self.m_leakvolt = [] #m3
        self.m_leakrad = 0.0 #m
        self.m_leakrate = 0.0 #m3/s
        
        #*** calculated paramters ***
        #exponential for N, alpha, gamma, and a
        r = np.random.exponential(scale=0.25,size=1)
        r[r>1.0] = 1.0
        r[r<0] = 0.0
        self.u_N = r[0]*(rock.N[2]-rock.N[0])+rock.N[0]        
        r = np.random.exponential(scale=0.25,size=1)
        r[r>1.0] = 1.0
        r[r<0] = 0.0
        self.u_alpha = r[0]*(rock.alpha[2]-rock.alpha[0])+rock.alpha[0]        
        r = np.random.exponential(scale=0.25,size=1)
        r[r>1.0] = 1.0
        r[r<0] = 0.0
        self.u_a = r[0]*(rock.a[2]-rock.a[0])+rock.a[0]        
        r = np.random.exponential(scale=0.25,size=1)
        r[r>1.0] = 1.0
        r[r<0] = 0.0
        self.u_gamma = r[0]*(rock.gamma[2]-rock.gamma[0])+rock.gamma[0]    
        #uniform for b
        r = np.random.uniform(rock.b[0],rock.b[2],size=1)
        self.u_b = r[0]
        #uniform for n1
        r = np.random.uniform(rock.n1[0],rock.n1[2],size=1)
        self.u_n1 = r[0]
        #exponential for bh
        r = np.random.exponential(scale=0.25,size=1)
        r[r>1.0] = 1.0
        r[r<0] = 0.0
        self.bh = r[0]*(rock.bh[2]-rock.bh[0])+rock.bh[0]
        #uniform for phi
        if phi < 0:
            r = np.random.uniform(rock.phi[0],rock.phi[2],size=1)
            self.phi = r[0]
        else:
            self.phi = phi
        #uniform for mcc
        if mcc < 0:
            r = np.random.uniform(rock.mcc[0],rock.mcc[2],size=1)
            self.mcc = r[0]
        else:
            self.mcc = mcc
        #stress state
        self.Pc, self.sn, self.tau = rock.stress.Pc_frac(self.str, self.dip, self.phi, self.mcc)
        #apertures
        self.bd = self.bh/self.u_N
        self.bd0 = self.bd
        self.vol = (4.0/3.0)*pi*0.25*self.dia**2.0*0.5*self.bd
            
       
#line objects
class line:
    def __init__(self,x0=0.0,y0=0.0,z0=0.0,length=1.0,azn=0.0*deg,dip=0.0*deg,w_type='pipe',dia=0.0254*3.0,rough=80.0):
        #position geometry
        self.c0 = np.asarray([x0, y0, z0]) #origin
        self.leg = length #length
        self.azn = azn #axis azimuth north
        self.dip = dip #axis dip from horizontal
        self.typ = typ(w_type) #type of well
        #flow geometry
        self.ra = dia #m
        self.rb = dia + 0.0254*0.5 # m
        self.rc = dia + 0.0254*1.0 # m
        self.rgh = 80.0

#node list object
class nodes:
    #initialization
    def __init__(self):
        self.r0 = np.asarray([np.inf,np.inf,np.inf])
        self.all = np.asarray([self.r0])
        self.tol = 0.001
        self.num = len(self.all)
        self.p = np.zeros(self.num,dtype=float)
        self.T = np.zeros(self.num,dtype=float)
        self.h = np.zeros(self.num,dtype=float)
#        self.f_id = [[-1]]
    #add a node
    def add(self,c=np.asarray([0.0,0.0,0.0])): #,f_id=-1):
        #round to within tolerance
        if not(np.isinf(c[0])):
            c = np.rint(c/self.tol)*self.tol
        #check for duplicate existing node
        ck_1 = np.asarray([np.isin(self.all[:,0],c[0]),np.isin(self.all[:,1],c[1]),np.isin(self.all[:,2],c[2])])
        ck_2 = ck_1[0,:]*ck_1[1,:]*ck_1[2,:]
        ck_i = np.where(ck_2 == 1)
        #yes duplicate -> return index of existing node
        if len(ck_i[0]) > 0:
#            if f_id != -1:
#                self.f_id[ck_i[0][0]] += [f_id]
            return False, ck_i[0][0]
        #no duplicate -> add node -> return index of new node
        else: 
            self.all = np.concatenate((self.all,np.asarray([c])),axis=0)
            self.num = len(self.all)
            self.p = np.concatenate((self.p,np.asarray([0.0])),axis=0)
            self.T = np.concatenate((self.T,np.asarray([0.0])),axis=0)
            self.h = np.concatenate((self.h,np.asarray([0.0])),axis=0)
#            self.f_id += [[f_id]]
            return True, len(self.all)-1

#pipe list object
class pipes:
    #initialization
    def __init__(self):
        self.num = 0
        self.n0 = [] #source node
        self.n1 = [] #target node
        self.L = [] #length
        self.W = [] #width
        self.typ = [] #property source type
        self.fID = [] #property source index
        self.K = [] #flow solver coefficient
        self.n = [] #flow solver exponent
    #add a pipe
    def add(self, n0, n1, length, width, featTyp, featID):
        self.n0 += [n0]
        self.n1 += [n1]
        self.L += [length]
        self.W += [width]
        self.typ += [featTyp]
        self.fID += [featID]
        self.K += [1.0]
        self.n += [1.0]
        self.num = len(self.n0)
        

    
#model object, functions, and data as object
class mesh:
    def __init__(self): #,node=[],pipe=[],fracs=[],wells=[],hydfs=[],bound=[],geo3D=[]): #@@@ are these still used?
#        #common reference geometry
#        geo = [None,None,None,None,None] #[natfracs, origin, intlines, points, wells]
#        geo[0]=sg.mergeObj(geo[0], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0.1,0,0]), r=0.1))
#        geo[1]=sg.mergeObj(geo[1], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([20,0,0]), r=3.0))
#        geo[1]=sg.mergeObj(geo[1], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0,20,0]), r=3.0))
#        geo[1]=sg.mergeObj(geo[1], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0,0,20]), r=3.0))
#        geo[2]=sg.mergeObj(geo[2], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0.1,0,0]), r=0.1))
#        geo[3]=sg.mergeObj(geo[3], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0.1,0,0]), r=0.1))
#        geo[4]=sg.mergeObj(geo[3], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0.1,0,0]), r=0.1))
        #domain information
        self.rock = reservoir()
        self.nodes = nodes()
        self.pipes = pipes()
        self.fracs = []
        self.wells = []
        self.hydfs = []
        self.bound = []
        self.faces = []
#        self.geo3D = geo #[natfracs, origin, intlines, points, wells]
        #intersections tracker
        self.trakr = [] #index of fractures in chain
        #flow solver
        self.H = [] #boundary pressure head array, m
        self.Q = [] #boundary flow rate array, m3/s
        self.q = [] #calculated pipe flow rates
#        self.bd = [] #mechanical aperture
#        self.bh = [] #hydraulic aperture
#        self.sn = [] #effective stress on fractures
        self.i_p = [] #constant flow well pressures, Pa
        self.i_q = [] #constant flow well rates, m3/s
        self.p_p = [] #constant pressure well pressures, Pa
        self.p_q = [] #constant pressure well rates, m3/s
        self.b_p = [] #boundary pressure, Pa
        self.b_q = [] #boundary rates, m3/s
        #heat solver
        self.Tb = [] #boundary temperatures
        self.R0 = [] #thermal radius
        self.Rt = [] #thermal radius over time
        self.ms = [] #pipe mass flow rates
        self.Et = [] #energy in rock over time
        self.Qt = [] #heat flow from rock over time
        self.Tt = [] #heat flow from rock over time
        self.ht = [] #enthalpy over time
        self.ts = [] #time stamps
        self.w_h = [] #wellhead enthalpy
        self.w_m = [] #wellhead mass flow
        self.p_E = [] #production energy
        self.b_h = [] #boundary enthalpy
        self.b_m = [] #boundary mass flow
        self.b_E = [] #boundary energy
        self.i_E = [] #injection energy
        self.p_mm = [] #mixed produced mass flow rate
        self.p_hm = [] #mixed produced enthalpy
        #power solver
        self.Pout = [] #rankine power output
        self.dhout = [] #heat extraction
        #validation stuff
        self.v_Rs = []
        self.v_ts = []
        self.v_Ps = []
        self.v_ws = []
        self.v_Vs = []
        self.v_Pn = []
#        #matrix flow solver
#        self.m_rh = [] #matrix hydraulic radius for each fracture
#        self.m_vh = [] #matrix volume gain for each fracture
        

#    # Static per-fracture hydraulic resistance terms using method of Luke P. Frash; [lower,nominal,upper]
#    # .... also calculates the stress states on the fractutres
#    def static_KQn(self,
#             rnd_N = [0.01,0.2,0.2],
#             rnd_alpha = [-0.002/MPa,-0.028/MPa,-0.080/MPa], 
#             rnd_a = [0.01,0.05,0.20], 
#             rnd_b = [0.7,0.8,0.9], 
#             rnd_gamma = [0.001,0.01,0.03]):
#        #size
#        num = len(self.faces)
#        #exponential for N, alpha, gamma, and a
#        r = np.random.exponential(scale=0.25,size=num)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        N = r*(rnd_N[2]-rnd_N[0])+rnd_N[0]        
#        r = np.random.exponential(scale=0.25,size=num)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        alpha = r*(rnd_alpha[2]-rnd_alpha[0])+rnd_alpha[0]        
#        r = np.random.exponential(scale=0.25,size=num)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        a = r*(rnd_a[2]-rnd_a[0])+rnd_a[0]        
#        r = np.random.exponential(scale=0.25,size=num)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        gamma = r*(rnd_gamma[2]-rnd_gamma[0])+rnd_gamma[0]        
#        #uniform for b
#        b = np.random.uniform(rnd_b[0],rnd_b[2],size=num)
#        #store properties
#        for n in range(0,num):
#            #aperture scaling parameters
#            self.faces[n].u_N = N[n]
#            self.faces[n].u_alpha = alpha[n]
#            self.faces[n].u_a = a[n]
#            self.faces[n].u_b = b[n]
#            self.faces[n].u_gamma = gamma[n]
#            #stress states
#            self.faces[n].Pc, self.faces[n].sn, self.faces[n].tau = self.rock.stress.Pc_frac(self.faces[n].str, self.faces[n].dip, self.faces[n].phi, self.faces[n].mcc)
            
    # L - d - M - k Gutenberg-Richter based aperture stimulation #@@@ can add ability to grow fractures here
    def GR_bh(self, f_id, fix=False): #, pp=-666.0):
        #initialize stim
        stim = False
        
        #check if fracture will be stimulated
        if (self.faces[f_id].Pmax >= self.faces[f_id].Pc):
            self.faces[f_id].stim += 1
            stim = True
            print('-> fracture stimulated: %i' %(f_id))

        #fracture parameters
        f_radius = 0.5*self.faces[f_id].dia
        
        #pressures for analysis
        e_max = self.faces[f_id].sn - self.faces[f_id].Pmax
        e_cen = self.faces[f_id].sn - self.faces[f_id].Pcen
        
        #if not stimulated
        if stim == False:
            #compute stress dependent apertures
            bd0 = self.faces[f_id].bd0
            bd = bd0*np.exp(self.faces[f_id].u_alpha*e_cen)
            bh = bd*self.faces[f_id].u_N
            #note that fracture is closed
            self.faces[f_id].hydroprop = False
        #if fixed (no stimulation allowed)
        elif fix:
            bd0 = self.faces[f_id].bd0
            #if open
            if (e_max < 0.0):
                #maximum aperture from Sneddon's (PKN) penny fracture aperture
                bd = (-8.0*e_max*(1.0-self.rock.Resv**2.0)*f_radius)/(pi*self.rock.ResE) #m
                #hydraulic aperture
                bh = bd*self.rock.f_roughness #m
                #add bd0 for equation continuity
                bd = bd + bd0
                bh = bh + bd0*self.faces[f_id].u_N
                #note that fracture is hydropropped
                self.faces[f_id].hydroprop = True
            #if closed
            else:
                #stress closure
                bd = bd0 * np.exp(self.faces[f_id].u_alpha*e_max)
                #hydraulic aperture
                bh = bd * self.faces[f_id].u_N
                #note that fracture is closed
                self.faces[f_id].hydroprop = False
        #if stimulated
        else:
            #*** shear component ***
            #G-R b-value for earthquake distribution
            bval = self.rock.bval
            
#            #maximum moment (potential slip method)
#            dnom = self.rock.gamma[1] * (2.0*f_radius) ** self.rock.n1[1]
#            dmax = self.rock.gamma[2] * (2.0*f_radius) ** self.rock.n1[2]
#            ddnomM = dmax-dnom
#            M0max = (pi*f_radius**2.0) * (0.5*ddnomM) * self.rock.ResG # circular displacement field, using max and nominal value
#            Mwmax = (np.log10(M0max)-9.1)/1.5
#            #minimum moment (matrix permeability method)
#            dbdnom = (12.0*self.rock.Porek/s_N**3.0)**0.5
#            dnomk = ((dbdnom + s_a * dnom ** s_b)/s_a)**(1/s_b)
#            ddnomk = dnomk-dnom
#            M0min = (pi*f_radius**2.0) * (0.5*ddnomk) * self.rock.ResG # circular displacement field, using nominal values only
#            Mwmin = (np.log10(M0min)-9.1)/1.5
            
            #maximum moment (shear stress method)
            M0max = self.faces[f_id].tau*(0.25*pi*self.faces[f_id].dia**2.0)**(3.0/2.0)
            Mwmax = (np.log10(M0max)-9.1)/1.5
            
            #minimum moment ('rule of 10')
            Mwmin = Mwmax - 1.0
            
            #sample Mw
            s_mw = np.random.exponential(1.0/(np.log(10)*bval)) + Mwmin
            #resample if out of range
            count = 0
            while 1:
                count += 1
                if s_mw > 1.0*Mwmax:
                    s_mw = np.random.exponential(1.0/(np.log(10)*bval)) + Mwmin
                else:
                    break
                if count > 100:
                    break
            #record stimulation magnitude
            self.faces[f_id].Mws += [s_mw]
            #convert to moment magnitude (Mo)
            s_mo = 10.0**(s_mw * 1.5 + 9.1)
            #convert to shear displacement
            s_ds = s_mo / ((pi*f_radius**2.0) * self.rock.ResG)
            #convert fracture length (L) to initial shear displacement (d0)
            d0 = 0.5 * self.faces[f_id].u_gamma * ((2.0*f_radius) ** self.faces[f_id].u_n1)
            #convert d0 to initial dilatant aperture (bd0)
            s_bd0 = self.faces[f_id].u_a * d0 ** self.faces[f_id].u_b
            #add stimulated displacement (s_ds) to initial displacement (d0)
            d1 = d0 + s_ds
            #convert new displacement (d1) to new dilatant aperture (bd1)
            s_bd1 = self.faces[f_id].u_a * d1 ** self.faces[f_id].u_b
            #calculate change in cumulative dilatant aperture (s_dbd)
            s_dbd = s_bd1 - s_bd0 
            #add to zero-stress dilatant aperture
            bd0 = self.faces[f_id].bd0 + s_dbd
            
            #*** tensile component ***
            #if open
            if (e_max < 0.0):
                #maximum aperture from Sneddon's (PKN) penny fracture aperture
                bd = (-8.0*e_max*(1.0-self.rock.Resv**2.0)*f_radius)/(pi*self.rock.ResE) #m
                #hydraulic aperture
                bh = bd*self.rock.f_roughness #m
                #add bd0 for equation continuity
                bd = bd + bd0
                bh = bh + bd0*self.faces[f_id].u_N
                #note that fracture is hydropropped
                self.faces[f_id].hydroprop = True
            #if closed
            else:
                #stress closure
                bd = bd0 * np.exp(self.faces[f_id].u_alpha*e_cen)
                #hydraulic aperture
                bh = bd * self.faces[f_id].u_N
                #note that fracture is closed
                self.faces[f_id].hydroprop = False
            
            #*** growth ***
            #grow fracture by larger of 20% fracture size or 5% domain size
            add_dia = np.max([0.2*self.faces[f_id].dia,0.05*self.rock.size])
            self.faces[f_id].dia += add_dia
                
        #limiters for flow solver stability
        if bh < self.rock.bh_min:
            bh = self.rock.bh_min
#            print('-> Alert: bh at min')
        elif bh > self.rock.bh_max:
            bh = self.rock.bh_max
            # print('-> Alert: bh at max')
        
        #override for boundary fractures
        if (int(self.faces[f_id].typ) in [typ('boundary')]):
            bh = self.rock.bh_bound
        
#        #reset stim
#        self.faces[f_id].stim = False
        
        #volume
        vol = (4.0/3.0)*pi*0.25*self.faces[f_id].dia**2.0*0.5*bd

        #update fracture properties
        self.faces[f_id].bd = bd
        self.faces[f_id].bh = bh
        self.faces[f_id].vol = vol
        self.faces[f_id].bd0 = bd0
        
        #return true if stimulated
        return stim
    
    def save(self,fname='input_output.txt',pin=''):
        out = []
        
        out += [['pin',pin]]
        
        #input parameters
        r = self.rock
        out += [['size',r.size]]
        out += [['ResDepth',r.ResDepth]]
        out += [['ResGradient',r.ResGradient]]
        out += [['ResRho',r.ResRho]]
        out += [['ResKt',r.ResKt]]
        out += [['ResSv',r.ResSv]]
        out += [['AmbTempC',r.AmbTempC]]
        out += [['AmbPres',r.AmbPres]]
        out += [['ResE',r.ResE]]
        out += [['Resv',r.Resv]]
        out += [['ResG',r.ResG]]
        out += [['Ks3',r.Ks3]]
        out += [['Ks2',r.Ks2]]
        out += [['s3Azn',r.s3Azn]]
        out += [['s3AznVar',r.s3AznVar]]
        out += [['s3Dip',r.s3Dip]]
        out += [['s3DipVar',r.s3DipVar]]
        for i in range(0,len(r.fNum)):
            out += [['fNum%i' %(i),r.fNum[i]]]
            out += [['fDia_min%i' %(i),r.fDia[i][0]]]
            out += [['fDia_max%i' %(i),r.fDia[i][1]]]
            out += [['fStr_nom%i' %(i),r.fStr[i][0]]]
            out += [['fStr_var%i' %(i),r.fStr[i][1]]]
            out += [['fDip_nom%i' %(i),r.fDip[i][0]]]
            out += [['fDip_var%i' %(i),r.fDip[i][1]]]
        for i in range(0,3):
            out += [['alpha%i' %(i),r.alpha[i]]]
        for i in range(0,3):
            out += [['gamma%i' %(i),r.gamma[i]]]
        for i in range(0,3):
            out += [['n1%i' %(i),r.n1[i]]]
        for i in range(0,3):
            out += [['a%i' %(i),r.a[i]]]
        for i in range(0,3):
            out += [['b%i' %(i),r.b[i]]]
        for i in range(0,3):
            out += [['N%i' %(i),r.N[i]]]
        for i in range(0,3):
            out += [['bh%i' %(i),r.bh[i]]]
        out += [['bh_min',r.bh_min]]
        out += [['bh_max',r.bh_max]]
        out += [['bh_bound',r.bh_bound]]
        out += [['f_roughness',r.f_roughness]]
        out += [['w_count',r.w_count]]
        out += [['w_spacing',r.w_spacing]]
        out += [['w_length',r.w_length]]
        out += [['w_azimuth',r.w_azimuth]]
        out += [['w_dip',r.w_dip]]
        out += [['w_proportion',r.w_proportion]]
        out += [['w_phase',r.w_phase]]
        out += [['w_toe',r.w_toe]]
        out += [['w_skew',r.w_skew]]
        out += [['w_intervals',r.w_intervals]]
        out += [['ra',r.ra]]
        out += [['rb',r.rb]]
        out += [['rc',r.rc]]
        out += [['rgh',r.rgh]]
        out += [['CemKt',r.CemKt]]
        out += [['CemSv',r.CemSv]]
        out += [['GenEfficiency',r.GenEfficiency]]
        out += [['LifeSpan',r.LifeSpan]]
        out += [['TimeSteps',r.TimeSteps]]
        out += [['p_whp',r.p_whp]]
        out += [['Tinj',r.Tinj]]
        out += [['H_ConvCoef',r.H_ConvCoef]]
        out += [['dT0',r.dT0]]
        out += [['dE0',r.dE0]]
        out += [['PoreRho',r.PoreRho]]
        out += [['Poremu',r.Poremu]]
        out += [['Porek',r.Porek]]
        out += [['Frack',r.Frack]]
        out += [['BH_T',r.BH_T]]
        out += [['BH_P',r.BH_P]]
        out += [['s1',r.s1]]
        out += [['s2',r.s2]]
        out += [['s3',r.s3]]
        out += [['perf',r.perf]]
        out += [['r_perf',r.r_perf]]
        out += [['sand',r.sand]]
        out += [['leakoff',r.leakoff]]
        out += [['dPp',r.dPp]]
        out += [['dPi',r.dPi]]
        out += [['stim_limit',r.stim_limit]]
        out += [['Qinj',r.Qinj]]
        out += [['Vinj',r.Vinj]]
        out += [['Qstim',r.Qstim]]
        out += [['Vstim',r.Vstim]]
        out += [['bval',r.bval]]
        for i in range(0,3):
            out += [['phi%i' %(i),r.phi[i]]]
        for i in range(0,3):
            out += [['mcc%i' %(i),r.mcc[i]]]
        out += [['hfmcc',r.hfmcc]]
        out += [['hfphi',r.hfphi]]
        
        #total injection rate (+ in)
        qinj = 0.0
        key = np.where(np.asarray(self.i_q)>0.0)[0]
        for i in key:
            qinj += self.i_q[i]
        key = np.where(np.asarray(self.p_q)>0.0)[0]
        for i in key:
            qinj += self.p_q[i]
        out += [['qinj',qinj]]
        
        #total production rate (-out)
        qpro = 0.0
        key = np.where(np.asarray(self.i_q)<0.0)[0]
        for i in key:
            qpro += self.i_q[i]
        key = np.where(np.asarray(self.p_q)<0.0)[0]
        for i in key:
            qpro += self.p_q[i]
        out += [['qpro',qpro]]
        
        #total leakoff rate (-out)
        qoff = 0.0
        key = np.where(np.asarray(self.b_q)<0.0)[0]
        for i in key:
            qoff += self.b_q[i]
        out += [['qleak',qoff]]
        
        #total boundary uptake (+in)
        qup = 0.0
        key = np.where(np.asarray(self.b_q)>0.0)[0]
        for i in key:
            qup += self.b_q[i]
        out += [['qgain',qup]]
        
        #recovery
        qrec = 0.0
        if qinj > 0:
            qrec = qpro/qinj
        else:
            qrec = 1.0
        out += [['recovery',qrec]]
        
        #largest quake
        quake = -10.0
        for i in range(0,len(self.faces)):
            # if self.faces[i].Mws != []:
            if self.faces[i].Mws:
                quake = np.max([quake,np.max(self.faces[i].Mws)])
        out += [['max_quake',quake]]

#        #flow results
#        for i in range(0,len(self.i_p)):
#            out += [['i_p%i' %(i),self.i_p[i]]]
#            out += [['i_q%i' %(i),self.i_q[i]]]
#        for i in range(0,len(self.p_p)):
#            out += [['p_p%i' %(i),self.p_p[i]]]
#            out += [['p_q%i' %(i),self.p_q[i]]]
#        for i in range(0,len(self.b_p)):
#            out += [['b_p%i' %(i),self.b_p[i]]]
#            out += [['b_q%i' %(i),self.b_q[i]]]
            
#        #identify injectors and producers
#        iPro = np.where(self.w_m<=0.0)[0]
##        iInj = np.where(self.w_m>0.0)[0]
#            
#        #temporal results
#        for i in iPro:
#            out += [['w_m%i' %(i),self.w_m[i]]]
#        for t in range(0,len(self.ts)-1):
#            for i in iPro:
#                out += [['p_h%i : %.3f y' %(i,self.ts[t]/yr),self.w_h[i][t]]]
#        out += [['p_mm',self.p_mm]]
#        for t in range(0,len(self.ts)-1):
#            out += [['p_hm : %.3f y' %(self.ts[t]/yr),self.p_hm[t]]]
#        out += [['tot_b_q',np.sum(self.b_q)]]
#        out += [['tot_b_m ',np.sum(self.b_m)]]
#        for t in range(0,len(self.ts)-1):
#            out += [['b_h : %.3f y' %(self.ts[t]/yr),self.b_h[0][t]]]
        
        #injection pressure
        pinj = np.max(np.asarray(self.p_p))/MPa
        out += [['pinj',pinj]]
        
        #injection enthalpy
        T5 = self.rock.Tinj + 273.15 # K
        if pinj > 100.0:
            pinj = 100.0
        state = therm(T=T5,P=pinj)
        hinj = state.h
        out += [['hinj',hinj]]
        
        #injection intercepts
        ixint = 0
        for i in range(0,self.pipes.num):
            if (int(self.pipes.typ[i]) in [typ('injector')]):
                ixint += 1
        ixint = ixint - self.rock.w_intervals
        out += [['ixint',ixint]]
        
        #production intercepts
        pxint = 0
        for i in range(0,self.pipes.num):
            if (int(self.pipes.typ[i]) in [typ('producer')]):
                pxint += 1
        pxint = pxint - self.rock.w_count
        out += [['pxint',pxint]]
        
        #stimulated fractures
        hfstim = 0
        nfstim = 0
        for i in range(0,len(self.faces)):
            if (self.faces[i].stim > 0):
                if (int(self.faces[i].typ) in [typ('propped')]):
                    hfstim += 1
                elif (int(self.faces[i].typ) in [typ('fracture')]):
                    nfstim += 1
        out += [['hfstim',hfstim]]
        out += [['nfstim',nfstim]]
        
        #production mass flow rate
        out += [['mpro',self.p_mm]]
        
        #production enthalpy over time
        for t in range(0,len(self.ts)-1):
            out += [['hpro:%.3f' %(self.ts[t]/yr),self.p_hm[t]]]
        
        #power
        Pout = []
        if not self.Pout:
            Pout = np.full(len(self.ts),np.nan)
        else:
            Pout = self.Pout
        for t in range(0,len(self.ts)-1):
            out += [['Pout:%.3f' %(self.ts[t]/yr),Pout[t]]]
            
        #thermal energy extraction
        dhout = []
        # if self.dhout == []:
        if not self.dhout.any():
            dhout = np.full(len(self.ts),np.nan)
        else:
            dhout = self.dhout
        for t in range(0,len(self.ts)-1):
            out += [['dhout:%.3f' %(self.ts[t]/yr),dhout[t]]]
        
        #output to file
        # out = zip(*out)
        out = list(map(list,zip(*out)))
        head = out[0][0]
        for i in range(1,len(out[0])):
            head = head + ',' + out[0][i]
        data = '%i' %(out[1][0])
        for i in range(1,len(out[1])):
            # if out[1][i] == []:
            if not out[1][i]:
                data = data + ',nan'
            else:
                data = data + ',%.5e' %(out[1][i])
        try:
            with open(fname,'r') as f:
                test = f.readline()
            f.close()
            if test != '':
                with open(fname,'a') as f:
                    f.write(data + '\n')
                f.close()
            else:
                with open(fname,'a') as f:
                    f.write(head + '\n')
                    f.write(data + '\n')
                f.close()
        except:
            with open(fname,'a') as f:
                f.write(head + '\n')
                f.write(data + '\n')
            f.close()
#        
#        return out
        
    def build_vtk(self,fname='default'):
        #******   paint wells   ******
        w_obj = [] #fractures
        w_col = [] #fractures colors
        w_lab = [] #fractures color labels
        w_lab = ['Well_Number','Well_Type','Inner_Radius','Roughness']
        w_0 = []
        w_1 = []
        w_2 = []
        w_3 = []
        #nodex = np.asarray(self.nodes)
        for i in range(0,len(self.wells)): #skip boundary node at np.inf
            #add colors
            w_0 += [i]
            w_1 += [self.wells[i].typ]
            w_2 += [self.wells[i].ra]
            w_3 += [self.wells[i].rgh]
            #add geometry
            azn = self.wells[i].azn
            dip = self.wells[i].dip
            leg = self.wells[i].leg
            vAxi = np.asarray([math.sin(azn)*math.cos(-dip), math.cos(azn)*math.cos(-dip), math.sin(-dip)])
            c0 = self.wells[i].c0
            c1 = c0 + vAxi*leg
            w_obj += [sg.cylObj(x0=c0, x1=c1, r=5.0)]
        #vtk file
        w_col = [w_0,w_1,w_2,w_3]
        sg.writeVtk(w_obj, w_col, w_lab, vtkFile=(fname + '_wells.vtk'))
        
        #******   paint fractures   ******
        f_obj = [] #fractures
        f_col = [] #fractures colors
        f_lab = [] #fractures color labels
        f_lab = ['Face_Number','Node_Number','Type','Sn_MPa','Pc_MPa']
        f_0 = []
        f_1 = []
        f_2 = []
        f_3 = []
        f_4 = []
        #nodex = np.asarray(self.nodes)
        for i in range(6,len(self.faces)): #skip boundary node at np.inf
            #add colors
            f_0 += [i]
            f_1 += [self.faces[i].ci]
            f_2 += [self.faces[i].typ]
            f_3 += [self.faces[i].sn/MPa]
            f_4 += [self.faces[i].Pc/MPa]
            #add geometry
            f_obj += [HF(r=0.5*self.faces[i].dia, x0=self.faces[i].c0, strikeRad=self.faces[i].str, dipRad=self.faces[i].dip, h=0.01)]
        #vtk file
        f_col = [f_0,f_1,f_2,f_3,f_4]
        sg.writeVtk(f_obj, f_col, f_lab, vtkFile=(fname + '_fracs.vtk'))
        
        #******   paint flowing fractures   ******
        q_obj = [] #fractures
        q_col = [] #fractures colors
        q_lab = [] #fractures color labels
        q_lab = ['Face_Number','Node_Number','Type','bd_mm','bh_mm','Sn_MPa','Pcen_MPa','Pc_MPa','stim','Pmax_MPa']
        q_0 = []
        q_1 = []
        q_2 = []
        q_3 = []
        q_4 = []
        q_5 = []
        q_6 = []
        q_7 = []
        q_8 = []
        q_9 = []
        #nodex = np.asarray(self.nodes)
        for i in range(6,len(self.faces)): #skip boundary node at np.inf
            if self.faces[i].ci >= 0:
                #add colors
                q_0 += [i]
                q_1 += [self.faces[i].ci]
                q_2 += [self.faces[i].typ]
                q_3 += [self.faces[i].bd*1000]
                q_4 += [self.faces[i].bh*1000]
                q_5 += [self.faces[i].sn/MPa]
                q_6 += [self.faces[i].Pcen]
                q_7 += [self.faces[i].Pc/MPa]
                q_8 += [self.faces[i].stim]
                q_9 += [self.faces[i].Pmax]
                #add geometry
                q_obj += [HF(r=0.5*self.faces[i].dia, x0=self.faces[i].c0, strikeRad=self.faces[i].str, dipRad=self.faces[i].dip, h=0.01)]
        #vtk file
        q_col = [q_0,q_1,q_2,q_3,q_4,q_5,q_6,q_7,q_8,q_9]
        sg.writeVtk(q_obj, q_col, q_lab, vtkFile=(fname + '_fnets.vtk'))

#        #******   paint stimulated fractures   ******
#        s_obj = [] #fractures
#        s_col = [] #fractures colors
#        s_lab = [] #fractures color labels
#        s_lab = ['Face_Number','Node_Number','Type','bd_mm','bh_mm','Sn_MPa','Pcen_MPa','Pc_MPa','stim','Pmax_MPa']
#        s_0 = []
#        s_1 = []
#        s_2 = []
#        s_3 = []
#        s_4 = []
#        s_5 = []
#        s_6 = []
#        s_7 = []
#        s_8 = []
#        s_9 = []
#        #nodex = np.asarray(self.nodes)
#        for i in range(6,len(self.faces)): #skip boundary node at np.inf
#            if self.faces[i].ci >= 0:
#                if self.faces[i].stim > 0:
#                    #add colors
#                    s_0 += [i]
#                    s_1 += [self.faces[i].ci]
#                    s_2 += [self.faces[i].typ]
#                    s_3 += [self.faces[i].bd*1000]
#                    s_4 += [self.faces[i].bh*1000]
#                    s_5 += [self.faces[i].sn/MPa]
#                    s_6 += [self.faces[i].Pcen]
#                    s_7 += [self.faces[i].Pc/MPa]
#                    s_8 += [self.faces[i].stim]
#                    s_9 += [self.faces[i].Pmax]
#                    #add geometry
#                    s_obj += [HF(r=0.5*self.faces[i].dia, x0=self.faces[i].c0, strikeRad=self.faces[i].str, dipRad=self.faces[i].dip, h=0.01)]
#        #vtk file
#        s_col = [s_0,s_1,s_2,s_3,s_4,s_5,s_6,s_7,s_8,s_9]
#        sg.writeVtk(s_obj, s_col, s_lab, vtkFile=(fname + '_fstim.vtk'))
        
        #******   paint nodes   ******
        n_obj = [] #nodes
        n_col = [] #nodes colors
        n_lab = [] #nodes color labels
        n_lab = ['Node_Number','Node_Pressure_MPa','Node_Temperature_K','Node_Enthalpy_kJ/kg']
        n_0 = []
        n_1 = []
        n_2 = []
        n_3 = []
        #nodex = np.asarray(self.nodes)
        for i in range(1,self.nodes.num): #skip boundary node at np.inf
            #add colors
            n_0 += [i]
            n_1 += [self.nodes.p[i]/MPa]
            n_2 += [self.nodes.T[i]]
            n_3 += [self.nodes.h[i]]
            #add geometry
            n_obj += [sg.cylObj(x0=self.nodes.all[i]+np.asarray([0.0,0.0,-3.0]), x1=self.nodes.all[i]+np.asarray([0.0,0.0,3.0]), r=3.0)]
        #vtk file
        n_col = [n_0,n_1,n_2,n_3]
        sg.writeVtk(n_obj, n_col, n_lab, vtkFile=(fname + '_nodes.vtk'))

        #******   paint pipes   ******
        p_obj = [] #pipes
        p_col = [] #pipes colors
        p_lab = [] #pipes color labels
        p_lab = ['Pipe_Number','Type','Pipe_Flow_Rate_m3_s','Height_m','Length_m']
        p_0 = []
        p_1 = []
        p_2 = []
        p_3 = []
        p_4 = []
        for i in range(0,self.pipes.num):
            #add geometry
            x0 = self.nodes.all[self.pipes.n0[i]]
            x1 = self.nodes.all[self.pipes.n1[i]]
            #don't include boundary node
            if not(np.isinf(x0[0]) or np.isinf(x1[0])):
                p_obj += [sg.cylObj(x0=x0, x1=x1, r=2.0)]            
                #add colors
                p_0 += [i]
                p_1 += [self.pipes.typ[i]]
                p_2 += [abs(self.q[i])]
                p_3 += [self.pipes.W[i]]
                p_4 += [self.pipes.L[i]]
        #vtk file
        p_col = [p_0,p_1,p_2,p_3,p_4]
        sg.writeVtk(p_obj, p_col, p_lab, vtkFile=(fname + '_flow.vtk'))               
        
#        #pipes and nodes vtk file
#        objectList=[self.geo3D[1], self.geo3D[2], self.geo3D[3]]
#        colorList= [0.25,     0.5,    0.75,]
#        sg.writeVtk(objectList,
#                    [colorList],
#                    ['Color'],
#                    vtkFile=(fname + '_pipes.vtk'))

    def build_pts(self,spacing=25.0,fname='test_gridx'):
        print('*** constructing temperature grid ***')
        #structured grid of datapoints
        fname = fname + '_therm.vtk'
        size = self.rock.size
        num = int(2.0*size/spacing)+1
        label = 'temp_K'
        ns = [num,num,num]
        o0 = [-size,-size,-size]
        ss = [spacing,spacing,spacing]
        
        #initialize data to initial rock temperature
        data = np.ones((num,num,num),dtype=float)*self.rock.BH_T
        
        #seek temperature drawdown
        for i in range(0,self.pipes.num):
            print('pipe %i' %(i))
            #collect fracture parameters
            x0 = self.nodes.all[self.pipes.n0[i]]
            x1 = self.nodes.all[self.pipes.n1[i]]
            T0 = self.nodes.T[self.pipes.n0[i]]
            T1 = self.nodes.T[self.pipes.n1[i]]
            c0 = 0.5*(x0+x1)
            r0 = 0.5*np.linalg.norm(x1-x0)
            R0 = self.R0[i]
            #pipes and wells
            if (int(self.pipes.typ[i]) in [typ('injector'),typ('producer'),typ('pipe')]): #((Y[i][2] == 0): #pipe, Hazen-Williams
                pass
            #fractures and planes
            elif (int(self.pipes.typ[i]) in [typ('fracture'),typ('propped'),typ('choke')]): #(int())Y[i][2] == 1: #fracture, effective cubic law
                #fracture info
                dip = self.faces[self.pipes.fID[i]].dip
                azn = self.faces[self.pipes.fID[i]].str
                vNor = np.asarray([math.sin(azn+90.0*deg)*math.sin(dip),math.cos(azn+90.0*deg)*math.sin(dip),math.cos(dip)])
                vLeg = (x1-c0)/np.linalg.norm(x1-c0)
                vWid = np.cross(vNor,vLeg) 
                #cycle thruogh all points
                for x in range(0,len(data)):
                    for y in range(0,len(data[0])):
                        for z in range(0,len(data[0,0])):
                            #point coordinates
                            xPt = np.asarray([o0[0]+x*spacing, o0[1]+y*spacing, o0[2]+z*spacing])
                            #spherical radius vector
                            pi = xPt-c0
                            #normal distance from fracture
                            ni = np.linalg.norm(np.dot(pi,vNor))
                            #lengthwise distance from fracture
                            li = np.linalg.norm(np.dot(pi,vLeg))
                            #widthwise distance from fracture
                            wi = np.linalg.norm(np.dot(pi,vWid))
                            #if within length, width, normal
                            if (ni <= R0) and (li <= r0) and (wi <= (0.5*self.pipes.W[i])):
                                #subtract delta T at point based on distance of point versus thermal radius
                                data[x,y,z] = data[x,y,z] + (0.5*(T1+T0)-self.rock.BH_T)*(1.0-ni/R0)

        head = '# vtk DataFile Version 2.0\n'
        head += 'pointcloud\n'
        head += 'ASCII\n'
        head += 'DATASET STRUCTURED_POINTS\n'
        head += 'DIMENSIONS %i %i %i\n' %(ns[0],ns[1],ns[2])
        head += 'ORIGIN %f %f %f\n' %(o0[0],o0[1],o0[2])
        head += 'SPACING %f %f %f\n' %(ss[0],ss[1],ss[2])
        head += 'POINT_DATA %i\n' %(ns[0]*ns[1]*ns[2])
        head += 'SCALARS ' + label + ' float 1\n'
        head += 'LOOKUP_TABLE default'
        
        print(head)
        
        try:
            with open(fname,'r') as f:
                test = f.readline()
            f.close()
            if test != '':
                print('file already exists')
                f.close()
            else:
                with open(fname,'a') as f:
                    f.write(head + '\n')
                    out = ''
                    for k in range(0,len(data[0,0,:])):
                        for j in range(0,len(data[0,:,0])):
                            for i in range(0,len(data[:,0,0])):
                                out += '%e' %(data[i,j,k]) + '\n'
                    f.write(out)
                f.close()
        except:
            with open(fname,'a') as f:
                f.write(head + '\n')
                out = ''
                for k in range(0,len(data[0,0,:])):
                    for j in range(0,len(data[0,:,0])):
                        for i in range(0,len(data[:,0,0])):
                            out += '%e' %(data[i,j,k]) + '\n'
                f.write(out)
            f.close()


    def re_init(self):
        #clear prior data
        self.nodes = []
        self.nodes = nodes()
        self.pipes = []
        self.pipes = pipes()
        self.faces = [] #all surfaces
        self.faces = self.bound + self.fracs + self.hydfs
        for i in range(0,len(self.faces)):
            self.faces[i].ci = -1
        #self.fstat = []
        #self.geo3D = geo3D
        self.trakr = []
        self.H = []
        self.Q = []
#        self.p = []
        self.nodes.p = self.nodes.p * 0.0
        self.nodes.T = self.nodes.T * 0.0
        self.nodes.h = self.nodes.h * 0.0
        self.q = []
        #self.bd = []
        #self.bh = []
        #self.sn = []
#        self.f2n = []
#        self.fp = []
        #self.static_KQn()
        return self
    
#    def set_f2n(self):
#        #build fracture ID to center node ID array
#        hold = -1*np.ones(len(self.faces),dtype=int)
#        for i in range(0,len(self.f2n)):
#            hold[int(self.f2n[i][0])] = self.f2n[i][1]
#        self.f2n = hold

    #flow solver boundary conditions with matrix flow considered
    def set_bcs(self,p_bound=0.0*MPa,q_well=[],p_well=[],matrix=True):
        #working variables
        rho = self.rock.PoreRho #kg/m3
        #input & output boundaries
        self.H = []
        self.Q = []
        #outer boundary (always zero index node)
        self.H += [[0, p_bound/(rho*g)]]
        #well boundaries (lists with empty elements)
        #q_well = [None,  None, 0.02,  None]
        #p_well = [None, 3.0e6, None, 1.0e6]
        #default = p_bound
        for w in range(0,len(self.wells)):
            #identify boundary node
            #coordinates
            source = self.wells[w].c0
            #find index of duplicate
            ck, i = self.nodes.add(source)
            if not(ck): #yes duplicate
                #prioritize flow boundary conditions
                if q_well[w] != None:
                    self.Q += [[i, -q_well[w]]]
                #pressure boundary conditions
                elif p_well[w] != None:
                    self.H += [[i, p_well[w]/(rho*g)]]
                #default to outer boundary condition if no boundary condition is explicitly stated
                else:
                    self.H += [[i, p_bound/(rho*g)]]
                    print('warning: a well was assigned the far-field pressure boundary condition')
            else:
                print('error: flow boundary point not identified')
        #matrix flow boundaries (based on boundary pressure)
        if matrix:
            #only include fractures that are part of the flow network
            for f in range(6,len(self.faces)): #skip boundary node at np.inf
                i = self.faces[f].ci
                if self.faces[f].ci >= 0:
                    Q = self.faces[f].m_leakrate
                    self.Q += [[i, Q]]
        
    
#    #matrix flow boundary conditions
#    def set_mbcs(self,p_bound=0.0*MPa,q_well=[],p_well=[]):
#        #working variables
#        rho = self.rock.PoreRho #kg/m3
#        #input & output boundaries
#        self.H = []
#        self.Q = []
#        #outer boundary (always zero index node)
#        self.H += [[0, p_bound/(rho*g)]]
#        #well boundaries (lists with empty elements)
#        #q_well = [None,  None, 0.02,  None]
#        #p_well = [None, 3.0e6, None, 1.0e6]
#        #default = p_bound
#        for w in range(0,len(self.wells)):
#            #identify boundary node
#            #coordinates
#            source = self.wells[w].c0
#            #find index of duplicate
#            ck, i = self.nodes.add(source)
#            if not(ck): #yes duplicate
#                #prioritize flow boundary conditions
#                if q_well[w] != None:
#                    self.Q += [[i, -q_well[w]]]
#                #pressure boundary conditions
#                elif p_well[w] != None:
#                    self.H += [[i, p_well[w]/(rho*g)]]
#                #default to outer boundary condition if no boundary condition is explicitly stated
#                else:
#                    self.H += [[i, p_bound/(rho*g)]]
#                    print('warning: a well was assigned the far-field pressure boundary condition')
#            else:
#                print('error: flow boundary point not identified')
        

#    def set_bcs(self,p_bound=0.0*MPa,q_inlet=[],q_outlet=[],p_inlet=[],p_outlet=[]):
#        #working variables
#        rho = self.rock.PoreRho #kg/m3
#        
#        #input & output boundaries
#        self.H = []
#        self.Q = []
#        
#        #outer boundary (always zero index node)
#        self.H += [[0, p_bound/(rho*g)]]
#        
#        #locate boundary conditions
#        n = [0,0,0,0]
#        for w in range(0,len(self.wells)):
#            #*** pressure boundary ***
#            #injection wells
#            if (self.wells[w].typ == typ('injector')) and (len(p_inlet)>0):
#                #coordinates
#                source = self.wells[w].c0
#                #find index of duplicate
#                ck, i = self.nodes.add(source)
#                if not(ck): #yes duplicate
#                    if len(p_inlet)>1:
#                        self.H += [[i, p_inlet[n[0]]/(rho*g)]]
#                        n[0] += 1
#                    else:
#                        self.H += [[i, p_inlet[0]/(rho*g)]]
#                else:
#                    print('error: flow boundary point not identified')
#            #production wells
#            if (self.wells[w].typ == typ('producer')) and (len(p_outlet)>0):
#                #coordinates
#                source = self.wells[w].c0
#                #find index of duplicate
#                ck, i = self.nodes.add(source)
#                if not(ck): #yes duplicate
#                    if len(p_outlet)>1:
#                        self.H += [[i, p_outlet[n[1]]/(rho*g)]]
#                        n[1] += 1
#                    else:
#                        self.H += [[i, p_outlet[0]/(rho*g)]]
#                else:
#                    print('error: pressure boundary point not identified')
#                
#            #*** flow boundary ***
#            #injection well
#            if (self.wells[w].typ == typ('injector')) and (len(q_inlet)>0):
#                #coordinates
#                source = self.wells[w].c0
#                #find index of duplicate
#                ck, i = self.nodes.add(source)
#                if not(ck): #yes duplicate
#                    if len(q_inlet)>1:
#                        self.Q += [[i, -q_inlet[n[2]]]]
#                        n[2] += 1
#                    else:
#                        self.Q += [[i, -q_inlet[0]]]
#                else:
#                    print('error: flow boundary point not identified')
#            #production wells
#            if (self.wells[w].typ == typ('producer')) and (len(q_outlet)>0):
#                #coordinates
#                source = self.wells[w].c0
#                #find index of duplicate
#                ck, i = self.nodes.add(source)
#                if not(ck): #yes duplicate
#                    if len(q_outlet)>1:
#                        self.Q += [[i, q_outlet[n[3]]]]
#                        n[3] += 1
#                    else:
#                        self.Q += [[i, q_outlet[0]]]
#                else:
#                    print 'error: pressure boundary point not identified'

    def therm_bcs(self,T_bound=0.0,T_inlet=[]):
#        #search array
#        narr = np.asarray(self.nodes)
        
        #outer boundary (always zero index node)
        self.Tb += [[0, T_bound]]
        
        #boundary conditions from wells
        n = 0
        for w in range(0,len(self.wells)):
            #temperature boundary condition
            if (self.wells[w].typ == typ('injector')):
                #coordinates
                source = self.wells[w].c0
                
                #find index of duplicate
                ck, i = self.nodes.add(source)
                if not(ck): #yes duplicate
                    if len(T_inlet)>1:
                        self.Tb += [[i, T_inlet[n]]]
                        n += 1
                    else:
                        self.Tb += [[i, T_inlet[0]]]
                else:
                    print('error: pressure temperature point not identified')
                
    def add_flowpath(self, source, target, length, width, featTyp, featID):
#        #ignore if source == target
#        if list(source) == list(target):
#            return -1, -1
        #source node
        n_s, so_n = self.nodes.add(source)#,f_id=featID)
        #target node
        n_t, ta_n = self.nodes.add(target)#,f_id=featID)
        #check if source is same as target
        if so_n == ta_n:
            return -1, -1
        #check if the reversed node set already exists (i.e., don't create pipes forward and backward between same nodes)
        if not(n_s) and not(n_t):
            to_ck = np.where(np.asarray(self.pipes.n0) == ta_n)[0]
            if len(to_ck) > 0:
                for i in to_ck:
                    if self.pipes.n1[i] == so_n:
                        return -1, -1
        #add pipe
        self.pipes.add(so_n, ta_n, length, width, featTyp, featID)
        return so_n, ta_n
        
    #intersections of lines with lines
    def x_well_wells():
        pass
        
        
        
    #intersections of a line with a plane
    def x_well_all_faces(self,plot=True,sourceID=0,targetID=[],offset=np.asarray([0.0,0.0,10.0])): #, path_type=0, aperture=0.22, roughness=80.0): #[x0,y0,zo,len,azn,dip]
        #working array for finding and logging intersection points
        x_well = [] #intercept coord
        o_frac = [] #surface origin
        r_frac = [] #index of well
        i_frac = [] #index of frac
        
        #line location
        c0 = self.wells[sourceID].c0 #line origin
        leg = self.wells[sourceID].leg #line length
        azn = self.wells[sourceID].azn #line azimuth
        dip = self.wells[sourceID].dip #line dip
        dia = self.wells[sourceID].ra #line inner diameter
        lty = self.wells[sourceID].typ #line type
        vAxi = np.asarray([math.sin(azn)*math.cos(-dip),math.cos(azn)*math.cos(-dip),math.sin(-dip)])
        cm = c0 + 0.5*leg*vAxi #line midpoint
        c1 = c0 + leg*vAxi #line endpoint
        
        #for all target faces
        for targetID in range(0,len(self.faces)):
            #planar object parameters
            t0 = self.faces[targetID].c0 #face origin
            rad = 0.5*self.faces[targetID].dia #face diameter
            azn = self.faces[targetID].str #face strike
            dip = self.faces[targetID].dip #face dip
            fty = self.faces[targetID].typ #face type
            #vDip = np.asarray([math.sin(azn+90.0*deg)*math.cos(-dip),math.cos(azn+90.0*deg)*math.cos(-dip),math.sin(-dip)])
            #vAzn = np.asarray([math.sin(azn),math.cos(azn),0.0])
            vNor = np.asarray([math.sin(azn+90.0*deg)*math.sin(dip),math.cos(azn+90.0*deg)*math.sin(dip),math.cos(dip)])
            #infinite plane intersection point
            if np.dot(vNor,vAxi) != 0: #not parallel
                x_test = c0 + vAxi*(np.dot(vNor,t0) - np.dot(vNor,c0))/(np.dot(vNor,vAxi))
            
                # test for intersect within plane and line extents
                if (np.linalg.norm(cm-x_test) < (0.5*leg)) and (np.linalg.norm(t0-x_test) < (rad)):
                    x_well += [x_test]
                    o_frac += [t0]
                    r_frac += [rad] 
                    i_frac += [targetID]
#                    fs_i += [targetID]
                    self.trakr += [[-1,targetID]]
#        #draw well geometry
#        if plot: #plot 3D
#            self.geo3D[4]=sg.mergeObj(self.geo3D[4], sg.cylObj(x0=c0, x1=c1, r=2.0))
        #in case of no intersections
        #add_flowpath(self, source, target, length, width, featTyp, featID, tol = 0.001):
        if len(x_well) == 0:
            #add endpoint to endpoint
            self.add_flowpath(c0,
                             c1 + offset,
                             leg,
                             dia,
                             lty,
                             sourceID)
            #return False
            return False
#            if plot: #plot 3D
#                self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=c0, x1=c1, r=3.0))
        #in case of intersections
        else:
            #convert to array #@@@@@@ this is new
            x_well = np.asarray(x_well)
            
            #sort intersections by distance from origin point
            rs = []
            rs = np.linalg.norm(x_well-c0,axis=1)
            a = rs.argsort()
            #first element well-well link (a live end)
            self.add_flowpath(c0,
                             x_well[a[0]] + offset,
                             rs[a[0]],
                             dia,
                             lty,
                             sourceID)
#            if plot: #plot 3D
#                self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=c0, x1=x_well[a[0]], r=3.0))
            #intersection points
            i = 0
            for i in range(0,len(rs)-1):
                #well-well links (+1.0 z offset to prevent non-real links from fracture to well without a choke)
                self.add_flowpath(x_well[a[i]] + offset,
                                  x_well[a[i+1]] + offset,
                                  rs[a[i+1]]-rs[a[i]],
                                  dia,
                                  lty,
                                  sourceID)
                #well-choke links (circumference of well * 3.0 * diameter = near well flow channel area dimensions, otherwise properties of the fracture)
                self.add_flowpath(x_well[a[i]] + offset,
                                 x_well[a[i]] + offset*0.5,
                                 3.0*dia,
                                 math.pi*dia,
                                 typ('choke'),
                                 i_frac[a[i]])
                #choke-fracture-center links (use intercept to center length, but fix width to y at 1/2 cirle radius)
                p_1, p_2 = self.add_flowpath(x_well[a[i]] + offset*0.5,
                                 o_frac[a[i]],
                                 np.linalg.norm(o_frac[a[i]]-(x_well[a[i]]+offset*0.5)),
                                 0.866*r_frac[a[i]],
                                 fty,
                                 i_frac[a[i]])
                #store fracture centerpoint node number
                if p_2 >= 0:
                    self.faces[i_frac[a[i]]].ci = p_2
#                    self.f2n += [[targetID,p_2]]
#                if plot: #plot 3D
#                    self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=x_well[a[i]], x1=x_well[a[i+1]], r=3.0))
#                    self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=x_well[a[i]], x1=o_frac[a[i]], r=3.0))
#                    self.geo3D[3]=sg.mergeObj(self.geo3D[3], sg.cylObj(x0=x_well[a[i]], x1=x_well[a[i]]+np.asarray([0.0,0.0,0.5]), r=3.0))
            #last segment well-choke link
            self.add_flowpath(x_well[a[-1]] + offset,
                             x_well[a[-1]] + offset*0.5,
                             3.0*dia,
                             math.pi*dia,
                             typ('choke'),
                             i_frac[a[-1]])
            #last segment choke-fracture link
            p_1, p_2 = self.add_flowpath(x_well[a[-1]] + offset*0.5,
                             o_frac[a[-1]],
                             np.linalg.norm(o_frac[a[-1]]-(x_well[a[-1]]+offset*0.5)),
                             0.866*r_frac[a[-1]],
                             fty,
                             i_frac[a[i]])
            #dead end segment
            self.add_flowpath(x_well[a[-1]] + offset,
                              c1,
                              np.linalg.norm(x_well[a[-1]]-c1), #,axis=1),
                              dia,
                              lty,
                              sourceID)
            #store fracture centerpoint node number
            if p_2 >= 0:
                self.faces[i_frac[a[-1]]].ci = p_2
#                self.f2n += [[targetID,p_2]]
#            #last well dead end segment can be ignored
#            if plot: #plot 3D
#                self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=x_well[a[-1]], x1=o_frac[a[-1]], r=3.0))
#                self.geo3D[3]=sg.mergeObj(self.geo3D[3], sg.cylObj(x0=x_well[a[-1]], x1=x_well[a[-1]]+np.asarray([0.0,0.0,0.5]), r=3.0))
            return True
    
    #intersections of a plane with a plane
    def x_frac_face(self,plot=True,sourceID=0,targetID=1): #[x0,y0,z0,dia,azn,dip]
        #plane 1
        dia1 = 0.5*self.faces[sourceID].dia
        dip = self.faces[sourceID].dip
        azn = self.faces[sourceID].str
        vNor1 = np.asarray([math.sin(azn+90.0*deg)*math.sin(dip),math.cos(azn+90.0*deg)*math.sin(dip),math.cos(dip)])
        c01 = self.faces[sourceID].c0
        f1_t = self.faces[sourceID].typ
#        f1_a = self.faces[sourceID,7]
#        f1_r = self.faces[sourceID,8]
#        if plot: #plot 3D fracture
#            fracture = HF(r=0.5*dia1,x0=c01,strikeRad=azn,dipRad=dip,h=0.01)
#            self.geo3D[0]=sg.mergeObj(self.geo3D[0], fracture)
        
        #plane 2
        dia2 = 0.5*self.faces[targetID].dia
        dip = self.faces[targetID].dip
        azn = self.faces[targetID].str
        vNor2 = np.asarray([math.sin(azn+90.0*deg)*math.sin(dip),math.cos(azn+90.0*deg)*math.sin(dip),math.cos(dip)])
        c02 = self.faces[targetID].c0
        f2_t = self.faces[targetID].typ
#        f2_a = self.faces[targetID,7]
#        f2_r = self.faces[targetID,8]
#        if plot: #plot 3D fracture
#            fracture = HF(r=0.5*dia2,x0=c02,strikeRad=azn,dipRad=dip,h=0.01)
#            self.geo3D[0]=sg.mergeObj(self.geo3D[0], fracture)
        
        #intersection vector
        vInt = []
        vInt = np.cross(vNor1,vNor2)
        
        #if not parallel
        if np.dot(vNor1,vNor2) < 0.999999:
            #intersection vector origin point
            zero = np.argmax(np.abs(np.asarray(vInt)),axis=0)
            d1 = -1*(vNor1[0]*c01[0] + vNor1[1]*c01[1] + vNor1[2]*c01[2])
            d2 = -1*(vNor2[0]*c02[0] + vNor2[1]*c02[1] + vNor2[2]*c02[2])
            vN1 = np.delete(vNor1, zero, axis=0)
            vN2 = np.delete(vNor2, zero, axis=0)
            cInt = (np.asarray([np.linalg.det([[vN1[1],vN2[1]],[d1,d2]]),
                   np.linalg.det([[d1,d2],[vN1[0],vN2[0]]])]) 
                   / np.linalg.det([[vN1[0],vN2[0]],[vN1[1],vN2[1]]]))
            cInt = np.insert(cInt,zero,0.0,axis=0)

            #endpoints - plane 1 intersection
            c = (cInt[0]-c01[0])**2.0 + (cInt[1]-c01[1])**2.0 + (cInt[2]-c01[2])**2.0 - dia1**2.0
            b = 2.0*( (cInt[0]-c01[0])*vInt[0] + (cInt[1]-c01[1])*vInt[1] + (cInt[2]-c01[2])*vInt[2] )
            a = vInt[0]**2.0 + vInt[1]**2.0 + vInt[2]**2.0
            if (b**2.0-4.0*a*c) >= 0.0:
                l1 = (-b+(b**2.0-4.0*a*c)**0.5)/(2.0*a)
                l2 = (-b-(b**2.0-4.0*a*c)**0.5)/(2.0*a)
                f1a = cInt + l1 * vInt
                f1b = cInt + l2 * vInt
            else:
                f1a = np.asarray([np.nan,np.nan,np.nan])
                f1b = np.asarray([np.nan,np.nan,np.nan])            
            
            #endpoints - plane 2 intersection
            c = (cInt[0]-c02[0])**2.0 + (cInt[1]-c02[1])**2.0 + (cInt[2]-c02[2])**2.0 - dia2**2.0
            b = 2.0*( (cInt[0]-c02[0])*vInt[0] + (cInt[1]-c02[1])*vInt[1] + (cInt[2]-c02[2])*vInt[2] )
            a = vInt[0]**2.0 + vInt[1]**2.0 + vInt[2]**2.0
            if (b**2.0-4.0*a*c) >= 0.0:
                l1 = (-b+(b**2.0-4.0*a*c)**0.5)/(2.0*a)
                l2 = (-b-(b**2.0-4.0*a*c)**0.5)/(2.0*a)
                f2a = cInt + l1 * vInt
                f2b = cInt + l2 * vInt
            else:
                f2a = np.asarray([np.nan,np.nan,np.nan])
                f2b = np.asarray([np.nan,np.nan,np.nan])
            
            #midpoint
            xInt = np.asarray([f1a,f1b,f2a,f2b])
            xInt = np.unique(xInt,axis=0)
            slot = len(xInt)-1
            xMid = []
            for i in range(0,slot+1):
                if (np.linalg.norm(xInt[slot-i]-c01) >= 1.01*(dia1)) or (np.linalg.norm(xInt[slot-i]-c02) >= 1.01*(dia2)) or (np.sum(np.isnan(xInt)) > 0):
                    xInt = np.delete(xInt,(slot-i),axis=0)
            if len(xInt) == 2 and np.sum(np.isnan(xInt)) == 0:
                xMid = 0.5*(xInt[0]+xInt[1])
#                self.geo3D[3]=sg.mergeObj(self.geo3D[3], sg.cylObj(x0=(xMid+np.asarray([0.0,0.0,-5.0])), x1=(xMid+np.asarray([0.0,0.0,5.0])), r=5.0))
    #            if plot: #plot 3D intersection line
    #                for i in xInt:
    #                    self.geo3D[3]=sg.mergeObj(self.geo3D[3], sg.cylObj(x0=(i+np.asarray([0.0,0.0,-5.0])), x1=(i+np.asarray([0.0,0.0,5.0])), r=5.0))
    #                if (np.sum(np.isnan(xInt)) == 0) and len(xInt) >= 2:
    #                    self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=xInt[0], x1=xInt[1], r=5.0))
            #add pipes to network            
            if (np.sum(np.isnan(xInt)) == 0) and len(xInt) == 2:
                #normal fracture-fracture connection
                if (f1_t != typ('boundary')) and (f2_t != typ('boundary')):
                    #source-center to intersection midpoint
                    #add_flowpath(self, source, target, length, width, featTyp, featID):
                    self.add_flowpath(c01,
                                     xMid,
                                     np.linalg.norm(xMid-c01),
                                     np.linalg.norm(xInt[1]-xInt[0]),
                                     f1_t,
                                     sourceID)
                    #intersection midpoint to target-center
                    p_1, p_2 = self.add_flowpath(xMid,
                                     c02,
                                     np.linalg.norm(xMid-c02),
                                     np.linalg.norm(xInt[1]-xInt[0]),
                                     f2_t,
                                     targetID)
                    #store fracture centerpoint node number
                    if p_2 >= 0:
                        self.faces[targetID].ci = p_2
#                        self.f2n += [[targetID,p_2]]
#                    #plot flow lines
#                    if plot:
#                        self.geo3D[3]=sg.mergeObj(self.geo3D[3], sg.cylObj(x0=(xMid+np.asarray([0.0,0.0,-3.0])), x1=(xMid+np.asarray([0.0,0.0,3.0])), r=3.0))
#                        self.geo3D[3]=sg.mergeObj(self.geo3D[3], sg.cylObj(x0=(c02+np.asarray([0.0,0.0,-3.0])), x1=(c02+np.asarray([0.0,0.0,3.0])), r=3.0))
#                        self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=c01, x1=xMid, r=3.0))
#                        self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=xMid, x1=c02, r=3.0))
                    #update tracker
                    self.trakr += [[sourceID,targetID]] 
                #fracture-boundary connection (boundary type = -3)
                elif (f2_t == typ('boundary')) and (f1_t != typ('boundary')):
                    #source-center to intersection midpoint
                    self.add_flowpath(c01,
                                     xMid,
                                     np.linalg.norm(xMid-c01),
                                     np.linalg.norm(xInt[1]-xInt[0]),
                                     f1_t,
                                     sourceID)
                    #intersection midpoint to far-field
                    p_1, p_2 = self.add_flowpath(xMid,
                                     self.nodes.r0,
                                     100.0*dia2,
                                     np.linalg.norm(xInt[1]-xInt[0]),
                                     f2_t,
                                     targetID)
                    #store fracture centerpoint node number
                    if p_2 >= 0:
                        self.faces[targetID].ci = p_2
#                        self.f2n += [[targetID,p_2]]
#                    print 'boundary intersection added'
#                    #plot flow lines
#                    if plot:
#                        self.geo3D[3]=sg.mergeObj(self.geo3D[3], sg.cylObj(x0=(xMid+np.asarray([0.0,0.0,-3.0])), x1=(xMid+np.asarray([0.0,0.0,3.0])), r=3.0))
#                        self.geo3D[3]=sg.mergeObj(self.geo3D[3], sg.cylObj(x0=(c02+np.asarray([0.0,0.0,-3.0])), x1=(c02+np.asarray([0.0,0.0,3.0])), r=3.0))
#                        self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=c01, x1=xMid, r=3.0))
#                        self.geo3D[2]=sg.mergeObj(self.geo3D[2], sg.cylObj(x0=xMid, x1=c02, r=3.0))
#                    #update tracker
#                    self.trakr += [[sourceID,targetID]]      
#        return self
    
#    #fracture pressure tracker
#    def update_face_pressure(self):
#        #initialize array if not yet created
#        if len(self.fp) == 0:
#            self.fp = np.zeros(len(self.faces))
#        #update fracture center point pressure
#        for f in range(0,len(self.faces)):
#            #set pressure
#            if self.f2n[f] >= 0:
#                self.fp[f] = self.p[int(self.f2n[f])] #Pa
                
    # ********************************************************************
    # domain creation
    # ********************************************************************
    def gen_domain(self,plot=True):
        print('*** domain boundaries module ***')
        #clear old data
        self.bound = []
        #working variables
        size = self.rock.size
        #create boundary faces for analysis
        domb3D = []
        domb3D += [surf(-size,0.0,0.0,4.0*size,00.0*deg,90.0*deg,'boundary',self.rock)]
        domb3D += [surf(size,0.0,0.0,4.0*size,00.0*deg,90.0*deg,'boundary',self.rock)]
        domb3D += [surf(0.0,-size,0.0,4.0*size,90.0*deg,90.0*deg,'boundary',self.rock)]
        domb3D += [surf(0.0,size,0.0,4.0*size,90.0*deg,90.0*deg,'boundary',self.rock)]
        domb3D += [surf(0.0,0.0,-size,4.0*size,00.0*deg,00.0*deg,'boundary',self.rock)]
        domb3D += [surf(0.0,0.0,size,4.0*size,00.0*deg,00.0*deg,'boundary',self.rock)]
        #add to model domain
        self.bound = domb3D        
        #plot 3D wireframe boundary
#        if plot: 
#            bp = np.asarray([[-size,-size,-size], [-size,-size,size], [-size,size,size], [size,size,size],
#                             [size,-size,size], [size,-size,-size], [size,size,-size], [-size,size,-size]])
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[0],x1=bp[1], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[1],x1=bp[2], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[2],x1=bp[3], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[3],x1=bp[4], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[4],x1=bp[5], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[5],x1=bp[6], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[6],x1=bp[7], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[7],x1=bp[0], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[0],x1=bp[5], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[1],x1=bp[4], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[3],x1=bp[6], r=2.0))
#            self.geo3D[1]=sg.mergeObj(self.geo3D[1], sg.cylObj(x0=bp[2],x1=bp[7], r=2.0))

    # ********************************************************************
    # fracture network creation
    # ********************************************************************
    def gen_fixfrac(self,clear=True,
                     c0 = [0.0,0.0,0.0],
                     dia = 500.0,
                     azn = 80.0*deg,
                     dip = 90.0*deg):
        print('-> manual fracture placement module')
        #clear old data
        if clear:
            self.fracs = []
        
        #place fracture
        c0 = np.asarray(c0)
        
        #compile list of fractures
        frac3D = [surf(c0[0],c0[1],c0[2],dia,azn,dip,'fracture',self.rock)]
        
        print('dia,azn,dip')
        print(dia)
        print(azn)
        print(dip)

        #add to model domain
        self.fracs += frac3D
    
    # ********************************************************************
    # fracture network creation
    # ********************************************************************
    def gen_natfracs(self,clear=True,
                     f_num = 40,
                     f_dia = [200.0,900.0],
                     f_azn = [79.0*deg,8.0*deg],
                     f_dip = [90.0*deg,12.5*deg]):
        print('-> dfn seed module')
        #clear old data
        if clear:
            self.fracs = []
        #size of domain for reservoir
        size = self.rock.size 
        #working variables
#        frac = []
        frac3D = []
        #populate fractures
        for n in range(0,f_num):
            #Fracture parameters
            dia = np.random.uniform(f_dia[0],f_dia[1])
            azn = np.random.normal(f_azn[0],f_azn[1])
            dip = np.random.normal(f_dip[0],f_dip[1])
            #Build geometry
            x = np.random.uniform(-size,size)
            y = np.random.uniform(-size,size)
            z = np.random.uniform(-size,size)
            c0 = np.asarray([x,y,z])
            #compile list of fractures
            frac3D += [surf(c0[0],c0[1],c0[2],dia,azn,dip,'fracture',self.rock)]
            
            #stuff for 2D imaging
            #vDip = np.asarray([math.sin(azn+90.0*deg)*math.cos(-dip),math.cos(azn+90.0*deg)*math.cos(-dip),math.sin(-dip)])
            #vAzn = np.asarray([math.sin(azn),math.cos(azn),0.0])
#            c1 = c0 + 0.5*dia*vAzn
#            c2 = c0 - 0.5*dia*vAzn
#            frac += [[c2,c1]]
            #Store fracture parameters; type = 1 for natural fracture; roughness (N) = 1.0 for smooth
            #[x0,yo,zo,leg,Azn,Dip,Type,Aperture,Roughness]
            
#            #plot 3D fracture
#            if plot: 
#                fracture = HF(r=0.5*dia,x0=c0,strikeRad=azn,dipRad=dip,h=0.01)
#                self.geo3D[0]=sg.mergeObj(self.geo3D[0], fracture)
        # Convert to arrays for slicing
#        frac = np.asarray(frac) #[[n],[a/b],[x/y/z]]
#        frac3D = np.asarray(frac3D) #[x0,y0,z0,dia,azn,dip]
        
        #add to model domain
        self.fracs += frac3D
    
    # ************************************************************************
    # well placement
    # ************************************************************************
    def gen_joint_sets(self):
        print('*** joint set module ***')
        #clear old joint sets
        self.fracs = []
        #generate fractures from sets
        for i in range(0,len(self.rock.fNum)):
            self.gen_natfracs(False,
                         self.rock.fNum[i],
                         self.rock.fDia[i],
                         self.rock.fStr[i],
                         self.rock.fDip[i])

    # ************************************************************************
    # stimulation - static
    # ************************************************************************
    def gen_stimfracs(self,plot=True,
                      target=0,
                      stages=1,
                      perfs=1,
                      f_dia = [1266.0,107.0],
                      f_azn = [65.0*deg,4.0*deg],
                      f_dip = [57.5*deg,3.75*deg],
                      clear = True):
        #print('*** stimulation module ***')
        #clear old data
        if clear == True:
            self.hydfs = []
        #hydraulic fractures
        num = stages*perfs
        #stimulate target well
        spa = self.wells[target].leg/(num+1)
        c0 = self.wells[target].c0
        azn = self.wells[target].azn
        dip = self.wells[target].dip
        vAxi = np.asarray([math.sin(azn)*math.cos(-dip),math.cos(azn)*math.cos(-dip),math.sin(-dip)])
        for n in range(0,num):
            #Fracture parameters
            leg = np.random.normal(f_dia[0],f_dia[1])
            azn = np.random.normal(f_azn[0],f_azn[1])
            dip = np.random.normal(f_dip[0],f_dip[1])
            c0 = c0 + spa*vAxi
            
            #add to model domain
            self.hydfs += [surf(c0[0],c0[1],c0[2],leg,azn,dip,'propped',self.rock, mcc = self.rock.hfmcc, phi = self.rock.hfphi)]
            self.hydfs[-1].bd0 = 0.0
    
    # ************************************************************************
    # well placement
    # ************************************************************************
    def gen_wells(self,clear=True,wells=[]):
        print('*** well placement module ***')
        #clear old data
        if clear:
            self.wells = []
        #manual placement
        if len(wells) > 0:
            self.wells = wells
        #automatic placement
        else:
            #center
            i0 = np.asarray([0.0, 0.0, 0.0])
            #ref axes (parallel injector and 90 horizontal to the right)
            azn = self.rock.w_azimuth
            dip = self.rock.w_dip
            vInj = np.asarray([math.sin(azn)*math.cos(-dip),
                     math.cos(azn)*math.cos(-dip),math.sin(-dip)])
            vRht = np.asarray([math.sin(azn+pi/2),math.cos(azn+pi/2),0.0])
            vNor = np.asarray([math.sin(azn)*math.sin(-dip),
                     math.cos(azn)*math.sin(-dip),math.cos(-dip)])
            #offset center
            p0 = i0 + vRht*self.rock.w_spacing
            #toe in production well
            toe = self.rock.w_toe
            vPro = sg.rotatePoints([vInj],vNor,toe)[0]
            #skew in production well
            skew = self.rock.w_skew
            vPro = sg.rotatePoints([vPro],vRht,skew)[0]
            #phase of production wells
            p0s = []
            vPros = []
            phase = self.rock.w_phase
            num = self.rock.w_count
            for i in range(0,num):
                p0s += [sg.rotatePoints([p0],vInj,(i*2.0*pi/num + phase))[0]]
                vPros += [sg.rotatePoints([vPro],vInj,(i*2.0*pi/num + phase))[0]]
            #lengths of wells
            length = self.rock.w_length
            proportion = self.rock.w_proportion
            iLen = length*proportion
            pLen = length
            #injection well segments
            i1s = []
            i2s = []
            seg = self.rock.w_intervals
            leg = iLen/(seg+seg-1)
            i1s += [i0 - 0.5*vInj*iLen]
            i2s += [i1s[0] + leg*vInj]
            for i in range(1,seg):
                i1s += [i2s[i-1] + leg*vInj]
                i2s += [i1s[i] + leg*vInj]
            #length of producers
            p1s = []
            p2s = []
            for i in range(0,num):
                p1s += [p0s[i] - 0.5*vPros[i]*pLen]
                p2s += [p0s[i] + 0.5*vPros[i]*pLen]
            #place injection wells
            wells = []
            azn, dip = azn_dip(i2s[0],i1s[0])
            for i in range(0,seg):
                wells += [line(i1s[i][0],i1s[i][1],i1s[i][2],leg,azn,dip,'injector',self.rock.ra,self.rock.rgh)]
            #place production wells
            for i in range(0,num):
                azn, dip = azn_dip(p2s[i],p1s[i])
                wells += [line(p1s[i][0],p1s[i][1],p1s[i][2],pLen,azn,dip,'producer',self.rock.ra,self.rock.rgh)]
            #add to model domain
            self.wells = wells

    # ************************************************************************
    # stimulation - add frac
    # ************************************************************************
    def add_frac(self,typ = 'propped',
                      c0 = np.asarray([0.0,0.0,0.0]),
                      dia = [1266.0,107.0],
                      azn = [65.0*deg,4.0*deg],
                      dip = [57.5*deg,3.75*deg]):
        print('   + placing new frac')
        fleg = np.random.normal(dia[0],dia[1])
        fazn = np.random.normal(azn[0],azn[1])
        fdip = np.random.normal(dip[0],dip[1])
        self.hydfs += [surf(c0[0],c0[1],c0[2],fleg,fazn,fdip,typ,self.rock)]
    
    # ************************************************************************
    # find intersections
    # ************************************************************************
    def gen_pipes(self,plot=True):
#        print('*** intersections module ***')
        #This code segment:
        #   1. finds intersections & break lines into segments for the flow
        #   2. builds the input deck for the flow model                              
        #   3. be compatible with heat transfer model
        #   4. not produce duplicate segments
        #   5. be as efficient as I can muster
        #   6. reject dead ends if possible
        #   7. add infinite fracture elements to nodes on boundary
#        #re-initialize the geometry - building list of faces
#        self.re_init()
        #calculate intersections for each well
        found = False
        for w in range(0,len(self.wells)):
            found += self.x_well_all_faces(sourceID=w)
        if found > 0:
            #chain intersections without repeating same comparitors
            iters = 0
            maxit = 20
            lock = 0
            hold = 0
            while 1:
                #loop breaker
                iters += 1
                if iters > maxit:
                    print('-> intersection search stopped after 20 iterations')
                    break
                #focus on chain connecting back to wells
                track = np.asarray(self.trakr)
                hold = len(self.trakr)
                conn = True
                #remove duplicate sources
                s_s = track[lock:,1]
                s_s = np.unique(s_s,axis=0)
                for s in s_s:
                    if s >= 0:
                        #search through all faces
                        for t in range(0,len(self.faces)):
                            #check if already searched
                            ck_1 = np.asarray([np.isin(track[:,0],t),np.isin(track[:,1],s)])
                            ck_2 = ck_1[0,:]*ck_1[1,:]
                            ck_3 = np.sum(ck_2)
                            #evaluate if value is valid
                            if (s != t) and (t >= 0) and (ck_3 == 0) and not(s in track[:,0]): #this pair is fresh, check for intersections
                                self.x_frac_face(plot=plot,sourceID=s,targetID=t)
                                conn = False
                        #update tracker
                        self.trakr += [[s,-1]]
                #lockout repeat searches
                lock = hold
                #break if no connections in this search
                if conn:
                    break
            #report number of iterations used to find intersections
            print('-> all intersections found using %i iters of %i allowable' %(iters,maxit))
        else:
            print('-> wells do not intersect any fractures')

    # ************************************************************************
    #energy generation - single flash steam rankine cycle
    # ************************************************************************
    def get_power(self,detail=False):
        #initialization
        self.Pout = []        
        #truncate pressures when superciritical
        i_p = np.max([list(self.i_p) + list(self.p_p)])/MPa
        if i_p > 100.0:
            i_p = 100.0
        #for each moment in time
        for t in range(0,len(self.p_hm)-1):
            # Surface Injection Well (5)
            T5 = self.rock.Tinj + 273.15 # K
            P5 = i_p # MPa
            state = therm(T=T5,P=P5)
            h5 = state.h # kJ/kg
            s5 = state.s # kJ/kg-K
            x5 = state.x # steam quality
            v5 = state.v # m3/kg
            # Undisturbed Reservoir (r)
            Tr = self.rock.BH_T #K
            Pr = self.rock.BH_P/MPa # MPa
            state = therm(T=Tr,P=Pr)
            hr = state.h # kJ/kg
            sr = state.s # kJ/kg-K
            xr = state.x # steam quality
            vr = state.v # m3/kg
            # Surface Production Well (2)
            P2 = self.rock.p_whp/MPa # MPa
            h2 = self.p_hm[t] #kJ/kg
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
            P3s = self.rock.AmbPres
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
            # Turbine Work
            w3s = h2s - h3s # kJ/kg
            # Pump Work
            w5s = v5*(np.max(list(self.i_p)+list(self.p_p))/MPa-P4s)*10**3 # kJ/kg
            w5l = v5*(np.max(list(self.i_p)+list(self.p_p))/MPa-P2l)*10**3 # kJ/kg
            # Net Work
            wNs = w3s - (w5s + w5l)
            # Energy production
            Pout = 0.0 #kW
            # If sufficient steam quality
            if x2 > 0:
                mt = -self.p_mm
                ms = mt*x2
                ml = mt*(1.0-x2)
                Vt = (v5*mt)*(1000*60) # L/min
                Pout = ms*wNs*self.rock.GenEfficiency #kW
            self.Pout += [Pout]

        #print details
        if detail:
            print('\n*** Rankine Cycle Thermal State Values ***')
            print(("Inject (5): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T5,P5,h5,s5,x5,v5)))
            print( ("Reserv (r,1): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(Tr,Pr,hr,sr,xr,vr)))
            print( ("Produc (2): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T2,P2,h2,s2,x2,v2)))    
            print( ("Turbi (2s): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T2s,P2s,h2s,s2s,x2s,v2s)))
            print( ("Brine (2l): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T2l,P2l,h2l,s2l,x2l,v2l)))
            print( ("Exhau (3s): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T3s,P3s,h3s,s3s,x3s,v3s)))
            print( ("Conde (4s): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T4s,P4s,h4s,s4s,x4s,v4s)))
            print( ("Turbine Specific Work = %.2f kJ/kg" %(w3s)))
            print( ("Turbine Pump Work = %.2f kJ/kg \nBrine Pump Work = %.2f kJ/kg" %(w5s,w5l)))
            print( ("Net Specifc Work Turbine = %.2f kJ/kg" %(wNs)))
            print( "Turbine Flow Rate = %.2f kg/s" %(ms))
            print( "Bypass Flow Rate = %.2f kg/s" %(ml))
            print( "Well Flow Rate = %.2f kg/s" %(mt))
            print( "Well Flow Rate = %.2f L/min" %(Vt))
            print( "Power at %.2f yr = %.2f kW" %(self.ts[-1]/yr,Pout))
            
#    # ************************************************************************
#    # flow network model
#    # ************************************************************************
##    def get_flow(self,p_bound=0.0*MPa,q_inlet=[],q_outlet=[],p_inlet=[],p_outlet=[],reinit=True):
#    def get_mflow(self,p_bound=0.0*MPa,q_well=[],p_well=[],reinit=True,useprior=False,matrix=True):
#        #reinitialize if the mesh has changed
#        if reinit:
#            #clear data from prior run
#            self.re_init()
#            #generate pipes
#            self.gen_pipes()
#        
#        #set boundary conditions (m3/s) (Pa); 10 kg/s ~ 0.01 m3/s for water
##        self.set_bcs(p_bound=p_bound,q_inlet=q_inlet,q_outlet=q_outlet,p_inlet=p_inlet,p_outlet=p_outlet)
#        self.set_bcs(p_bound=p_bound,q_well=q_well,p_well=p_well)
#        
#        #get fluid properties
#        mu = self.rock.Poremu #cP
#        rho = self.rock.PoreRho #kg/m3
#        
#        #flow solver working variables
#        N = self.nodes.num
#        Np = self.pipes.num
#        H = self.H
#        Q = self.Q
#            
#        #initial guess for nodal pressure head
#        if useprior and not(reinit):
#            h = self.nodes.p/(rho*g)
#        else:
#            h = 1.0*np.random.rand(N) + p_bound/(rho*g)
#        q = np.zeros(N)
#        
#        #install boundary condition
#        for i in range(0,len(H)):
#            h[H[i][0]] = H[i][1]
#        for i in range(0,len(Q)):
#            q[Q[i][0]] = Q[i][1]
#        
#        #hydraulic resistance equations
#        K = np.zeros(Np)
#        n = np.zeros(Np)
#        
#        #stabilizing limiters
#        zlim = 0.3*self.rock.s3/(rho*g)
#        hup = (self.rock.s1+10.0*MPa)/(rho*g)
#        hlo = -101.4/(rho*g)
#        
#        #convergence
#        goal = 0.0005
#        
#        #hydraulic resistance terms
#        for i in range(0,Np):
#            #working variables
#            u = self.pipes.fID[i]
#            
#            #pipes and wells
#            if (int(self.pipes.typ[i]) in [typ('injector'),typ('producer'),typ('pipe')]): #((Y[i][2] == 0): #pipe, Hazen-Williams
#                K[i] = 10.7*self.pipes.L[i]/(self.wells[u].rgh**1.852*self.wells[u].ra**4.87) #metric (m)
#                n[i] = 1.852
#
#            #fractures and planes
#            elif (int(self.pipes.typ[i]) in [typ('boundary'),typ('fracture'),typ('propped'),typ('choke')]): #(int())Y[i][2] == 1: #fracture, effective cubic law
#                K[i] = (12.0*mu*self.pipes.L[i])/(rho*g*self.pipes.W[i]*self.faces[u].bh**3.0)
#                n[i] = 1.0                    
#            #porous media
#            elif (int(self.pipes.typ[i]) in [typ('darcy')]):
#                K[i] = mu*self.pipes.L[i]/(rho*g*self.faces[u].bd*self.pipes.W[i]*self.rock.Frack)
#                n[i] = 1.0
#            #type not handled
#            else:
#                print 'error: undefined type of conduit'
#                exit
#        #record info
#        self.pipes.K = K
#        self.pipes.n = n
#    
#        #iterative Newton-Rhapson solution to solve flow
#        iters = 0
#        max_iters = 100
#        z = h
#        while 1:
#            #loop breaker
#            iters += 1
#            if iters > max_iters:
#                print '-> Flow solver halted after %i iterations' %(iters-1)
#                break
#            elif np.max(np.abs(z)) < goal: #np.max(np.abs(z/(h+z))) < goal:
#                print '-> Flow solver converged to <%.2f%% of max head using %i iterations' %(goal*100.0,iters-1)
#                break
#
#            #re-initialize working variables
#            F = np.zeros(N)
#            F += q
#            D = np.zeros((N,N))
#            
#            #add jitters
#            h += 0.001*np.random.rand(N)
#            
#            #build matrix equations
#            for i in range(0,Np):
#                #working variables
#                n0 = self.pipes.n0[i]
#                n1 = self.pipes.n1[i]
#                
#                #Node flow equations
#                R = np.sign(h[n0]-h[n1])*(abs((h[n0]-h[n1]))/K[i])**(1.0/n[i])
#                F[n0] += R
#                F[n1] += -R
#                
#                #Jacobian (first derivative of the inflow-outflow equations)
#                if abs((h[n0]-h[n1])) == 0:
#                    J = 1.0
#                else:
#                    J = ((1.0/n[i])/K[i])*(abs((h[n0]-h[n1]))/K[i])**(1.0/n[i]-1.0)
#                D[n0,n0] += J
#                D[n1,n1] += J
#                D[n0,n1] += -J
#                D[n1,n0] += -J
#            
#            #remove defined boundary values
#            for i in range(0,len(H)):
#                F = np.delete(F,H[-1-i][0],0)
#                D = np.delete(D,H[-1-i][0],0)
#                D = np.delete(D,H[-1-i][0],1)
#            
#            #solve matrix equations
#            z = solve(D[:,:],F[:])
#            
#            #apply correction limiters to prevent excess overshoot
#            z[z > zlim] = zlim
#            z[z < -zlim] = -zlim
#            
#            #update pressures
#            for i in range(0,len(H)):
#                z = np.insert(z,H[i][0],0,axis=0)
#            h = h - 0.7*z #0.7 scaling factor seems to give faster convergence by reducing solver overshoot
#            
#            #apply physical limits
#            h[h > hup] = hup
#            h[h < hlo] = hlo
#        
#        #flow rates
#        q = np.zeros(Np)
#        for i in range(0,Np):
#            #working variables
#            n0 = self.pipes.n0[i]
#            n1 = self.pipes.n1[i]
#            #flow rate
#            q[i] = np.sign(h[n0]-h[n1])*(abs(h[n0]-h[n1])/K[i])**(1.0/n[i])
#            
#        #record results in class
#        self.nodes.p = h*rho*g
#        self.q = q
#
#        #collect well rates and pressures
#        i_q = []
#        i_p = []
#        p_q = np.zeros(len(self.wells),dtype=float)
#        p_p = np.zeros(len(self.wells),dtype=float)
#        
#        for w in range(0,len(self.wells)):
#            #coordinates
#            source = self.wells[w].c0
#            #find index of duplicate
#            ck, i = self.nodes.add(source)
#            #record pressure
#            p_p[w] = self.nodes.p[i]
#            #record flow rate
#            i_pipe = np.where(np.asarray(self.pipes.n0) == i)[0][0]
#            p_q[w] = self.q[i_pipe]
#                
#        #collect boundary rates and pressures
#        b_nodes = [0]
#        b_pipes = []
#        b_q = []
#        b_p = []
#        w = b_nodes[0]
#        b_pipes = np.where(np.asarray(self.pipes.n1) == w)[0]
#        b_p += [self.nodes.p[w]]
#        if len(b_pipes) > 0:
#            for w in b_pipes:
#                b_q += [-self.q[w]]
#        b_p = list(np.ones(len(b_q),dtype=float)*b_p[0])
#                
#        #store key well and boundary flow data
#        self.i_p = i_p
#        self.i_q = i_q
#        self.p_p = p_p
#        self.p_q = p_q
#        self.b_p = b_p
#        self.b_q = b_q

    # ************************************************************************
    # flow network model
    # ************************************************************************
#    def get_flow(self,p_bound=0.0*MPa,q_inlet=[],q_outlet=[],p_inlet=[],p_outlet=[],reinit=True):
    def get_flow(self,p_bound=0.0*MPa,q_well=[],p_well=[],reinit=True,useprior=False,matrix=True):
        #reinitialize if the mesh has changed
        if reinit:
            #clear data from prior run
            self.re_init()
            #generate pipes
            self.gen_pipes()
        
        #get fluid properties
        mu = self.rock.Poremu #cP
        rho = self.rock.PoreRho #kg/m3    
        
        #flow solver working variables
        N = self.nodes.num
        Np = self.pipes.num
        
        #initial guess for nodal pressure head and initialize pipe flow array
        if useprior and not(reinit):
            #prior node pressures
            h = self.nodes.p/(rho*g)
            #update matrix flow terms
            for f in range(0,len(self.faces)): #for i in range(6,len(self.faces)): #skip boundary node at np.inf
                #center node index
                i = self.faces[f].ci
                #update leak rates using new node pressures
                dP = self.nodes.p[i]-p_bound
                self.faces[f].m_leakrad = np.abs((self.faces[f].m_leakvolt[-1]/(0.25*pi*self.faces[f].dia**2.0))/(self.rock.m_Ct*dP))
                self.faces[f].m_leakrate = 0.5*pi*(self.faces[f].dia**2.0)*self.rock.Porek*dP/(self.rock.Poremu*self.faces[f].m_leakrad)
            
        else:
            #random node pressures
            h = 1.0*np.random.rand(N) + p_bound/(rho*g)
            self.nodes.p = h*rho*g
            #get initial matrix-flow parameters (+ for leakoff, + for flow out of node)
            dt = self.rock.LifeSpan/self.rock.TimeSteps
            dV0 = ((2.0*self.rock.Porek*self.rock.m_Ct*self.rock.m_dP0**2.0*dt)/(self.rock.Poremu))**0.5 #m3/m2
            if (self.rock.m_Ct*self.rock.m_dP0) != 0:
                rh0 = np.abs(dV0/(self.rock.m_Ct*self.rock.m_dP0))
            else:
                rh0 = 100.0
            #initial matrix-flow estimation
            for f in range(0,len(self.faces)): #for i in range(6,len(self.faces)): #skip boundary node at np.inf
                #center node index
                i = self.faces[f].ci
                #record initial conditions
                self.faces[f].m_leakrad = rh0
                self.faces[f].m_leakvol0 = dV0*0.25*pi*self.faces[f].dia**2.0
                self.faces[f].m_leakvolt += [self.faces[f].m_leakvol0]
                self.faces[f].m_leakrate = 0.5*pi*(self.faces[f].dia**2.0)*self.rock.Porek*(self.nodes.p[i]-p_bound)/(self.rock.Poremu*self.faces[f].m_leakrad)            
        
        #initialize pipe flow rates to zero
        q = np.zeros(N)

        #set boundary conditions (m3/s) (Pa); 10 kg/s ~ 0.01 m3/s for water
        self.set_bcs(p_bound=p_bound,q_well=q_well,p_well=p_well,matrix=matrix)
        H = self.H
        Q = self.Q
        #install boundary condition
        for i in range(0,len(H)):
            h[H[i][0]] = H[i][1]
        for i in range(0,len(Q)):
            q[Q[i][0]] = Q[i][1]
        
        
        
        #hydraulic resistance equations
        K = np.zeros(Np)
        n = np.zeros(Np)
        
        #stabilizing limiters
        zlim = 0.3*self.rock.s3/(rho*g)
        hup = (self.rock.s1+10.0*MPa)/(rho*g)
        hlo = -101.4/(rho*g)
        
        #convergence
        goal = 0.0005
        
        #hydraulic resistance terms
        for i in range(0,Np):
            #working variables
            u = self.pipes.fID[i]
            
            #pipes and wells
            if (int(self.pipes.typ[i]) in [typ('injector'),typ('producer'),typ('pipe')]): #((Y[i][2] == 0): #pipe, Hazen-Williams
                K[i] = 10.7*self.pipes.L[i]/(self.wells[u].rgh**1.852*self.wells[u].ra**4.87) #metric (m)
                n[i] = 1.852

            #fractures and planes
            elif (int(self.pipes.typ[i]) in [typ('boundary'),typ('fracture'),typ('propped'),typ('choke')]): #(int())Y[i][2] == 1: #fracture, effective cubic law
                K[i] = (12.0*mu*self.pipes.L[i])/(rho*g*self.pipes.W[i]*self.faces[u].bh**3.0)
                n[i] = 1.0                    
            #porous media
            elif (int(self.pipes.typ[i]) in [typ('darcy')]):
                K[i] = mu*self.pipes.L[i]/(rho*g*self.faces[u].bd*self.pipes.W[i]*self.rock.Frack)
                n[i] = 1.0
            #type not handled
            else:
                print('error: undefined type of conduit')
                exit
        #record info
        self.pipes.K = K
        self.pipes.n = n
    
        #iterative Newton-Rhapson solution to solve flow
        iters = 0
        max_iters = 100
        z = h
        while 1:
            #loop breaker
            iters += 1
            if iters > max_iters:
                print('-> Flow solver halted after %i iterations' %(iters-1))
                break
            elif np.max(np.abs(z)) < goal: #np.max(np.abs(z/(h+z))) < goal:
                print('-> Flow solver converged to <%.2f%% of max head using %i iterations' %(goal*100.0,iters-1))
                break
            
            #update matrix flow terms #@@@@@@@@@@@@@@@@@
            for f in range(0,len(self.faces)): #for i in range(6,len(self.faces)): #skip boundary node at np.inf
                #center node index
                i = self.faces[f].ci
                #update leak rates using new node pressures
                dP = self.nodes.p[i]-p_bound
                self.faces[f].m_leakrad = np.abs((self.faces[f].m_leakvolt[-1]/(0.25*pi*self.faces[f].dia**2.0))/(self.rock.m_Ct*dP))
                self.faces[f].m_leakrate = 0.5*pi*(self.faces[f].dia**2.0)*self.rock.Porek*dP/(self.rock.Poremu*self.faces[f].m_leakrad)
            #@@@@@@@@@@@@@@@@@
            
            #re-install boundary conditions (m3/s) (Pa); 10 kg/s ~ 0.01 m3/s for water
            self.set_bcs(p_bound=p_bound,q_well=q_well,p_well=p_well,matrix=matrix)
            H = self.H
            Q = self.Q
            #install boundary condition
            for i in range(0,len(H)):
                h[H[i][0]] = H[i][1]
            for i in range(0,len(Q)):
                q[Q[i][0]] = Q[i][1]
                
            #re-initialize working variables
            F = np.zeros(N)
            F += q
            D = np.zeros((N,N))
            
            #add jitters
            h += 0.001*np.random.rand(N)
            
            #build matrix equations
            for i in range(0,Np):
                #working variables
                n0 = self.pipes.n0[i]
                n1 = self.pipes.n1[i]
                
                #Node flow equations
                R = np.sign(h[n0]-h[n1])*(abs((h[n0]-h[n1]))/K[i])**(1.0/n[i])
                F[n0] += R
                F[n1] += -R
                
                #Jacobian (first derivative of the inflow-outflow equations)
                if abs((h[n0]-h[n1])) == 0:
                    J = 1.0
                else:
                    J = ((1.0/n[i])/K[i])*(abs((h[n0]-h[n1]))/K[i])**(1.0/n[i]-1.0)
                D[n0,n0] += J
                D[n1,n1] += J
                D[n0,n1] += -J
                D[n1,n0] += -J
            
            #remove defined boundary values
            for i in range(0,len(H)):
                F = np.delete(F,H[-1-i][0],0)
                D = np.delete(D,H[-1-i][0],0)
                D = np.delete(D,H[-1-i][0],1)
            
            #solve matrix equations
            z = solve(D[:,:],F[:])
            
            #apply correction limiters to prevent excess overshoot
            z[z > zlim] = zlim
            z[z < -zlim] = -zlim
            
            #update pressures
            for i in range(0,len(H)):
                z = np.insert(z,H[i][0],0,axis=0)
            h = h - 0.7*z #0.7 scaling factor seems to give faster convergence by reducing solver overshoot
            
            #apply physical limits
            h[h > hup] = hup
            h[h < hlo] = hlo
            
            #record results in nodes
            self.nodes.p = h*rho*g
        
        #flow rates
        q = np.zeros(Np)
        for i in range(0,Np):
            #working variables
            n0 = self.pipes.n0[i]
            n1 = self.pipes.n1[i]
            #flow rate
            q[i] = np.sign(h[n0]-h[n1])*(abs(h[n0]-h[n1])/K[i])**(1.0/n[i])
            
        #record results in class
        self.nodes.p = h*rho*g
        self.q = q

        #collect well rates and pressures
        i_q = []
        i_p = []
        p_q = np.zeros(len(self.wells),dtype=float)
        p_p = np.zeros(len(self.wells),dtype=float)
        
        for w in range(0,len(self.wells)):
            #coordinates
            source = self.wells[w].c0
            #find index of duplicate
            ck, i = self.nodes.add(source)
            #record pressure
            p_p[w] = self.nodes.p[i]
            #record flow rate
            i_pipe = np.where(np.asarray(self.pipes.n0) == i)[0][0]
            p_q[w] = self.q[i_pipe]
                
        #collect boundary rates and pressures
        b_nodes = [0]
        b_pipes = []
        b_q = []
        b_p = []
        w = b_nodes[0]
        b_pipes = np.where(np.asarray(self.pipes.n1) == w)[0]
        b_p += [self.nodes.p[w]]
        if len(b_pipes) > 0:
            for w in b_pipes:
                b_q += [-self.q[w]]
        b_p = list(np.ones(len(b_q),dtype=float)*b_p[0])
                
        #store key well and boundary flow data
        self.i_p = i_p
        self.i_q = i_q
        self.p_p = p_p
        self.p_q = p_q
        self.b_p = b_p
        self.b_q = b_q
    
    # ************************************************************************
    # heat transfer model - mod 1-28-2021
    # ************************************************************************
    def get_heat(self,plot=True,
                 t_n = -1, #steps
                 t_f = -1.0*yr, #s
                 H = -1.0, #kW/m2-K
                 dT0 = -666.6, #K
                 dE0 = -666.6): #kJ/m2
        print('*** heat flow module ***')
        #****** default parameters ******
        if t_n < 0:
            t_n = self.rock.TimeSteps
        if t_f < 0:
            t_f = self.rock.LifeSpan
        if H < 0:
            H = self.rock.H_ConvCoef
        if dT0 < -666.0:
            dT0 = self.rock.dT0
        if dE0 < -666.0:
            dE0 = self.rock.dE0
        
        #****** boundary parameters ******
        #truncate pressures when superciritical
        i_p = np.max([list(self.i_p) + list(self.p_p)])/MPa
        if i_p > 100.0:
            i_p = 100.0        
        # Surface Injection Well (5)
        T5 = self.rock.Tinj + 273.15 # K
        P5 = i_p # MPa
        state = therm(T=T5,P=P5)
        h5 = state.h # kJ/kg
        s5 = state.s # kJ/kg-K
        x5 = state.x # steam quality
        v5 = state.v # m3/kg
        print(("Inject (5): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(T5,P5,h5,s5,x5,v5)))
        # Undisturbed Reservoir (r)
        Tr = self.rock.BH_T #K
        Pr = self.rock.BH_P/MPa # MPa
        state = therm(T=Tr,P=Pr)
        hr = state.h # kJ/kg
        sr = state.s # kJ/kg-K
        xr = state.x # steam quality
        vr = state.v # m3/kg
        print( ("Reserv (r): T= %.2f; P= %.2f; h= %.2f, s= %.4f, x= %.4f, v= %.6f" %(Tr,Pr,hr,sr,xr,vr)))
        
        #****** enthalpy function linearization *******
        if xr > 0.0001:
            print( 'warning: reservoir water is mixed phase (x = %.3f) so solver will be unreliable' %(xr))
        x = np.linspace(T5,Tr,100,dtype=float)
        y = np.zeros(100,dtype=float)
        for i in range(0,len(x)):
            y[i] = therm(T=x[i],P=Pr).h #kJ/kg
        hTP = np.polyfit(x,y,3)
        ThP = np.polyfit(y,x,3)
        ##error checking
        #y2 = hTP[0]*x**3.0 + hTP[1]*x**2.0 + hTP[2]*x**1.0 + hTP[3]
        #x2 = ThP[0]*y**3.0 + ThP[1]*y**2.0 + ThP[2]*y**1.0 + ThP[3]
        #fig = pylab.figure(figsize=(8.0, 6.0), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
        #ax1 = fig.add_subplot(111)
        #ax1.plot(x,y,label='raw')
        #ax1.plot(x,y2,label='fitted')
        #ax1.plot(x2,y,label='reverse')
        #ax1.set_ylabel('Enthalpy (kJ/kg)')
        #ax1.set_xlabel('Temperature (K)')
        #ax1.legend(loc='upper left', prop={'size':8}, ncol=2, numpoints=1)     

        #****** explicit solver setup ******
        #working variables
        ts = np.linspace(0.0,t_f,t_n+1)
        dt = ts[1]-ts[0]
        #set boundary conditions #K
        self.therm_bcs(T_bound=Tr,
                       T_inlet=[T5])
        #set solver parameters
        N = self.nodes.num
        Np = self.pipes.num
        Tb = self.Tb
#        Y = self.pipes
        #initial guess for temperatures
        Tn = np.ones(N) * Tr #K
        #node pressures from flow solution
        Pn = np.asarray(self.nodes.p)/MPa #MPa
        #truncate high pressures for solver stability (supercritical states)
        Pn[Pn > 100.0] = 100.0 
        #install boundary condition
        for i in range(0,len(Tb)):
            Tn[Tb[i][0]] = Tb[i][1]
            
        #more working variables
        Lp = np.asarray(self.pipes.L)
        ms = np.asarray(self.q)/v5 #assume density of fluid as injection fluid density at wellhead
        hn = np.zeros(N)
        R0 = np.zeros(Np) #ones(len(Y))*2.0*BoreOR #initialize 'thermal radius' to 2x radius of borehole
        ResSv = self.rock.ResSv
        ResKt = self.rock.ResKt
        CemKt = self.rock.CemKt
        
        #memory working variables
        ht = np.zeros((t_n+1,N),dtype=float)
        Tt = np.zeros((t_n+1,N),dtype=float)
        Rt = np.zeros((t_n+1,Np),dtype=float)
        Et = np.zeros((t_n+1,Np),dtype=float)
        Qt = np.zeros((t_n+1,Np),dtype=float)
        Er = np.zeros(Np,dtype=float)
        
        #well geometry #@@@@@@@ need to update for a per-borehole basis
        Ris = self.rock.ra
        Ric = self.rock.rb
        Rir = self.rock.rc
        
        #functional form of energy vs thermal radius for specified borehole diameter (per unit DT and dL)
        Ror = np.linspace(1,2000,2000)#(BoreOR,200,100)
        ERor = 2*pi*ResSv*1.0*((1-np.log(Rir)/np.log(Rir/Ror))*(Ror**2-Rir**2)/2
            + (1/(2*np.log(Rir/Ror)))*(Ror**2*(np.log(Ror)-0.5)-Rir**2*(np.log(Rir)-0.5)))
        lnRor = np.log(Ror[:])
        lnERor = np.log(ERor[:])
        ERm, ERb, r_value, p_value, std_err = stats.linregress(lnERor,lnRor)  # E[j] = np.exp((np.log(Ror)-ERb)/ERm)*dL*(Tor-T[j]) # kJ
        
        #key indecies for convergence criteria (i.e., boundary nodes)
        key = []
        for e in self.H:
            key += [e[0]]
            key += [e[0]+1]
        for e in self.Q:
            key += [e[0]]
            key += [e[0]+1]
        
        #initial rock energy withdraw and heat transfer rates (to capture early cooling from drilling and flow tests, this also stabilizes the early time solution)
        for p in range(0,Np):
            #working variables
            #u = self.pipes.fID[p]
            dT = dT0
            #@@@@@ override initial timestep energy to impose stability
            dT = abs(Tr-T5)
            a = 1.0/(self.rock.ResSv*dT*self.rock.ResKt*10**-3)
            b = 1.0/self.rock.H_ConvCoef
            c = -2.0*dT*dt
            dE0 = (-b + (b**2.0 - 4.0*a*c)**0.5)/(2.0*a)
            dE0 = dE0 * 1.0
            self.rock.dE0 = dE0
            
            if (int(self.pipes.typ[p]) in [typ('injector'),typ('producer'),typ('pipe')]): #pipe, radial heat flow
                #initial rock energy withdraw
                Et[0,p] = dE0*2.0*pi*Rir*Lp[p] #kJ
                #initial rock thermal radius
                R0[p] = np.exp(ERm*(np.log(np.abs(Et[0,p]/(Lp[p]*dT))))+ERb) + Rir #m
                #initial rock energy transfer rates
                Qt[0,p] = Lp[p]/(1.0/(2.0*pi*Ris*H) + np.log(R0[p]/Rir)/(2.0*pi*ResKt*10**-3) + np.log(Rir/Ric)/(2.0*pi*CemKt*10**-3)) #kJ/K-s
            elif (int(self.pipes.typ[p]) in [typ('boundary'),typ('fracture'),typ('propped'),typ('darcy'),typ('choke')]): #fracture, plate heat flow
                #initial rock energy withdraw
                Et[0,p] = dE0*self.pipes.W[p]*Lp[p] #kJ
                #initial rock thermal radius
                R0[p] = Et[0,p]/(ResSv*self.pipes.W[p]*Lp[p]*dT) #m
                #initial rock energy transfer rates
                Qt[0,p] = (2.0*self.pipes.W[p]*Lp[p])/(1.0/(H) + R0[p]/(ResKt*10**-3)) #kJ/K-s
            else:
                print('error: segment type %s not identified' %(typ(int(self.pipes.typ[p]))))
    
        #convergence criteria
        goal = 0.5
        
        #iterate over time
        for t in range(0,len(ts)-1):
            #calculate nodal temperatures by pipe flow and nodal mixing
            iters = 0
            max_iters = 30
            err = np.ones(N)*goal*10
            while 1:
                #calculate nodal fluid enthalpy #@@@ requires consideration of sensible heating (0<X<1) if not supercritical
                hn = hTP[0]*Tn**3.0 + hTP[1]*Tn**2.0 + hTP[2]*Tn**1.0 + hTP[3]
                    
                kerr = np.max(np.abs(err))
    
                #loop breaker
                iters += 1
                if iters > max_iters:
                    print( 'Heat solver halted after %i iterations' %(iters-1))
                    break
                elif kerr < goal:#np.max(np.abs(err)) < goal:
                    print( 'Heat solver converged to %e Kelvin after %i iterations' %(goal,iters-1))
                    break
                
                #follow the flow to estimate heating/cooling of fluid per pipe
                Ap = np.zeros(Np,dtype=float)
                Bp = np.zeros(Np,dtype=float)
                Cp = np.zeros(Np,dtype=float)
                Dp = np.zeros(Np,dtype=float)
                Ep = np.zeros(Np,dtype=float)
                hm = np.zeros(N,dtype=float)
                mi = np.zeros(N,dtype=float)
                for p in range(0,Np):
                    #working variables
                    #u = self.pipes.fID[p]
                    n0 = self.pipes.n0[p]
                    n1 = self.pipes.n1[p]                    
                    
                    #equilibrium enthalpy
#                    heq0 = therm(T=Tr,P=Pn[n0]).h
#                    heq1 = therm(T=Tr,P=Pn[n1]).h
                    heq0 = hTP[0]*Tr**3.0 + hTP[1]*Tr**2.0 + hTP[2]*Tr**1.0 + hTP[3]
                    heq1 = heq0
                    
                    #working variables
                    Kp = 0.0
                    Qp = 0.0
#                    dh = 0.0

                    #positive flow
                    if ms[p] > 0:
                        #non-equilibrium conduction limited heating
                        Ap[p] = Qt[t,p]*(Tr - 0.5*(Tn[n1] + Tn[n0]))
                        #equilibrium conduction limited heating
                        Bp[p] = Qt[t,p]*(Tr - 0.5*(Tr + Tn[n0]))
                        #flow limited cooling to equilibrium
                        Cp[p] = ms[p]*(heq1 - hn[n0])
                        #flow limited cooling to non-equilibrium
                        Dp[p] = ms[p]*(hn[n1] - hn[n0])
    
                    #negative flow    
                    else:
                        #non-equilibrium conduction limited heating
                        Ap[p] = Qt[t,p]*(Tr - 0.5*(Tn[n0] + Tn[n1]))
                        #equilibrium conduction limited heating
                        Bp[p] = Qt[t,p]*(Tr - 0.5*(Tr + Tn[n1]))
                        #flow limited cooling to equilibrium
                        Cp[p] = -ms[p]*(heq0 - hn[n1])
                        #flow limited cooling to non-equilibrium
                        Dp[p] = -ms[p]*(hn[n0] - hn[n1])
                        
                    #take maximum of conduction terms because this will drive conduction heat deliverability
                    Kp = np.asarray([Ap[p],Bp[p]])[np.argmax(np.abs([Ap[p],Bp[p]]))]
                    #if flow limits heat extraction from rock, it will go to equilibrium
                    Qp = Cp[p]
                    #get the limiting term
                    KorQ = np.argmin(np.abs([Kp,Qp]))
                    Ep[p] = np.asarray([Kp,Qp])[np.argmin(np.abs([Kp,Qp]))]
                    
                    #conduction limited
                    if KorQ == 0:
                        #positive flow
                        if ms[p] > 0:
                            mi[n1] += ms[p]
                            hm[n1] += Ep[p] + ms[p]*hn[n0]
                        #negative flow
                        else:
                            mi[n0] += -ms[p]
                            hm[n0] += Ep[p] + -ms[p]*hn[n1]
                    #flow limited
                    else:
                        #positive flow
                        if ms[p] > 0:
                            mi[n1] += ms[p]
                            hm[n1] += ms[p]*heq1
                        #negative flow
                        else:
                            mi[n0] += -ms[p]
                            hm[n0] += -ms[p]*heq0
                
                #calculate nodal temperatures
                z = []
                z = np.zeros(N,dtype=float) #-0.0001*np.random.rand(N) + Tr #K
                hu = np.zeros(N,dtype=float)
                for n in range(0,N):
                    #mixed inflow enthalpy
                    if mi[n] > 0:
                        hu[n] = hm[n]/mi[n]
                    else:
                        hu[n] = hr
                    #calculate temperature at new enthalpy
#                    z[n] = therm(h=hu[n],P=Pn[n]).T
                    z[n] = ThP[0]*hu[n]**3.0 + ThP[1]*hu[n]**2.0 + ThP[2]*hu[n]**1.0 + ThP[3]
                    
                
                #install boundary condition
                for j in range(0,len(Tb)):
                    z[Tb[j][0]] = Tb[j][1]
#                    hu[Tb[j][0]] = therm(T=z[Tb[j][0]],P=Pn[Tb[j][0]]).h
                    hu[Tb[j][0]] = hTP[0]*z[Tb[j][0]]**3.0 + hTP[1]*z[Tb[j][0]]**2.0 + hTP[2]*z[Tb[j][0]]**1.0 + hTP[3]
                    
                #calculate error
                err = [] #np.zeros(N,dtype=float)
                err = Tn - z
                
                #update Tn
                Tn = z
                
            #store ht
            ht[t] = hu
            #store temperatures
            Tt[t,:] = Tn
            #store thermal radii
            Rt[t,:] = R0
    
            #extracted energy during this time step
            Et[t+1] = Et[t] + np.abs(Ep)*dt
            #rock energy tracker (added or lost)
            Er += Ep*dt
            for i in range(0,Np):
                #working variables
                n0 = self.pipes.n0[i]
                n1 = self.pipes.n1[i]
                dT = abs(Tr-0.5*(Tn[n1]+Tn[n0]))
                
                #@@@@ stabilizer
                dT = np.max([dT,dT0])

                #thermal radius for next time step
                if (int(self.pipes.typ[i]) in [typ('injector'),typ('producer'),typ('pipe')]): #pipe, radial heat flow
                    if (dT > 0) and (Et[t+1,i] > 0):
                        R0[i] = np.exp(ERm*(np.log(np.abs(Et[t+1,i]/(Lp[i]*dT))))+ERb) + Rir #+2.0*Rir # m
                    #Et[0,i] = np.exp((np.log(R0[i])-ERb)/ERm)*Lp[i]*(Tr-0.5*(Tn[Y[i][1]]+Tn[Y[i][0]])) # kJ
                    Qt[t+1,i] = Lp[i]/(1.0/(2.0*pi*Ris*H) + np.log(R0[i]/Rir)/(2.0*pi*ResKt*10**-3) + np.log(Rir/Ric)/(2.0*pi*CemKt*10**-3)) # kJ/K-s # Note converted Kt in W/m-K to kW/m-K
                    
                elif (int(self.pipes.typ[i]) in [typ('boundary'),typ('fracture'),typ('propped'),typ('darcy'),typ('choke')]): #fracture, plate heat flow
                    if (dT > 0) and (Et[t+1,i] > 0):
                        R0[i] = Et[t+1,i]/(ResSv*self.pipes.W[i]*Lp[i]*dT)
                    #Et[0,i] = ResSv*R0[i]*Y[i][4]*Lp[i]*(Tr-0.5*(Tn[Y[i][1]]+Tn[Y[i][0]])) # kJ
                    Qt[t+1,i] = (2.0*self.pipes.W[i]*Lp[i])/(1.0/(H) + R0[i]/(ResKt*10**-3)) # kJ/K-s # Note converted Kt in W/m-K to kW/m-K
                else:
                    print( 'error: segment type %s not identified' %(typ(int(self.pipes.typ[i]))))
        
        #store results
        self.Et = Et
        self.Qt = Qt
        self.nodes.T = Tn
        self.nodes.h = hn
        self.ms = ms
        self.Tt = Tt
        self.ht = ht
        self.R0 = R0
        self.Rt = Rt
        
        #parameters of interest
        #**********************
        #collect well temp, rate, enthalpy
        w_h = []
        w_T = []
        w_m = []
        for w in range(0,len(self.wells)):
            #coordinates
            source = self.wells[w].c0
            #find index of duplicate
            ck, i = self.nodes.add(source)
            #record temperature
            w_T += [Tt[:,i]]
            #record enthalpy
            w_h += [ht[:,i]]
            #record mass flow rate
            i_pipe = np.where(np.asarray(self.pipes.n0) == i)[0][0]
            w_m += [self.q[i_pipe]/v5]
        w_h = np.asarray(w_h)
        w_T = np.asarray(w_T)
        w_m = np.asarray(w_m)
            
        #identify injectors and producers
        iPro = np.where(w_m<=0.0)[0]
        iInj = np.where(w_m>0.0)[0]
        
        #boundary flows
        b_nodes = [0]
        b_pipes = []
        b_h = [] 
        b_T = []
        b_m = []
        w = b_nodes[0]
        b_pipes = np.where(np.asarray(self.pipes.n1) == w)[0]
        b_h += [ht[:,w]]
        b_T += [Tt[:,w]]
        if len(b_pipes) > 0:
            for w in b_pipes:
                b_m += [self.q[w]/v5]
        
        #total production energy
        p_ET = 0.0
        for i in iPro:
            i = int(i)
            p_ET += np.sum(w_m[i]*w_h[i]) #kJ/s
        p_ET = p_ET*dt/t_f #/len(p_h[w]) #kJ/s avg over all
        
        #mixed produced enthalpy and mass flow rate
        p_mm = []
        p_hm = np.zeros(len(ht))
        p_mm = 0.0
        for i in iPro:
            i = int(i)
            p_mm += w_m[i]
            p_hm += w_h[i]*w_m[i]
        p_hm = p_hm/p_mm
        self.p_mm = p_mm #mixed produced mass flow rate
        self.p_hm = p_hm #mixed produced enthalpy
                
        #total injection energy
        i_ET = 0.0
        for i in iInj:
            i = int(i)
            i_ET += np.sum(w_m[i]*w_h[i]) #kJ/s
        i_ET = i_ET*dt/t_f #/len(i_h[w])
        
        #total boundary energy flow
        b_ET_out = 0.0
        b_ET_in = 0.0
        b_ET = 0.0
        b_ni = 0
        b_no = 0
        if len(b_pipes) > 0:
            for w in range(0,len(b_pipes)):
                #outflow
                if b_m[w] > 0:
                    b_ET_out += -np.sum(b_m[w]*b_h[0])
                    b_no += 1
                #inflow
                else:
                    b_ET_in += -np.sum(b_m[w]*b_h[0])
                    b_ni += 1
            if b_no > 0:
                b_ET_out = b_ET_out*dt/t_f #/b_no
            if b_ni > 0:
                b_ET_in = b_ET_in*dt/t_f #/b_ni
            b_ET = b_ET_out + b_ET_in
    
        #rock energy for thermal radius
        #E_tot = (np.sum(Et[-1,:]) - np.sum(Et[0,:]))/t_f #kJ/s
        
        #rock energy change
        E_roc = np.sum(Er)/t_f #kJ/s
        
        #net system energy (less error in model if closer to zero)
        E_net = E_roc + p_ET + i_ET + b_ET #kJ/s
        print( '\nNet System Energy (kJ/s):' )
        print(E_net)
        
        #save key values
        self.ts = ts
        self.w_h = w_h
        self.w_m = w_m
        self.b_h = b_h
        self.b_m = b_m
        
        #calculate power output
        self.get_power(detail=False)
        
        #thermal energy extraction
        dht = np.zeros(len(ts),dtype=float)
        for i in range(0,len(w_m)):
            dht += -w_m[i]*w_h[i]
        self.dhout = dht
    
        ##print key results
        #print '\nNode Temperature (R)'
        #print Tn
        #print '\nNode Enthalpy (kJ/kg)'
        #print hn

        if plot: #plots
            fig = pylab.figure(figsize=(11.0, 8.5), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
            font = {'family' : 'serif',   'size'   : 16}
            pylab.rc('font', serif='Arial')
            pylab.rc('font', **font)
            ax1 = fig.add_subplot(221)
            for i in iPro:
                i = int(i)
                ax1.plot(ts[:-1]/yr,w_h[i][:-1],linewidth=1.5)
            ax1.set_xlabel('Time (yr)')
            ax1.set_ylabel('Production Enthalpy (kJ/kg)')
            ax1.set_ylim(bottom=0.0)
            ax1 = fig.add_subplot(222)
            ax1.plot(ts[:-1]/yr,self.Pout[:],linewidth=1.5,color='red')
            ax1.set_xlabel('Time (yr)')
            ax1.set_ylabel('Electrical Power Output (kW)')
            ax1.set_ylim(bottom=0.0)
            ax1 = fig.add_subplot(223)
            for i in iPro:
                i = int(i)
                ax1.plot(ts[:-1]/yr,w_T[i][:-1],linewidth=1.5)
            ax1.set_xlabel('Time (yr)')
            ax1.set_ylabel('Production Temperature (K)')
            ax1.set_ylim(bottom=273.0)
#            ax1.plot(ts/yr,np.sum(Et,axis=1),linewidth=0.5,color='black')
#            ax1.set_xlabel('Time (yr)')
#            ax1.set_ylabel('Rock Energy (kJ)')
            ax1 = fig.add_subplot(224)
            ax1.plot(ts[:-1]/yr,dht[:-1],linewidth=1.5,color='green')
            ax1.set_xlabel('Time (yr)')
            ax1.set_ylabel('Thermal Recovery (kJ/s)')
            ax1.set_ylim(bottom=0.0)

    # ************************************************************************
    # stimulation - add frac (Vinj is total volume per stage, sand ratio to frac slurry by volume)
    # ************************************************************************
    def dyn_stim(self,Vinj = -1.0, Qinj = -1.0, dpp = -666.6*MPa,
                      sand = -1.0, leakoff = -1.0,
                      target=0, perfs=-1, r_perf=-1.0,
                      visuals = True, fname = 'stim'):
        print( '*** dynamic stim module ***')
        #fetch defaults
        if perfs < 0:
            perfs = self.rock.perf
        if r_perf < 0:
            r_perf = self.rock.r_perf
        if sand < 0:
            sand = self.rock.sand
        if leakoff < 0:
            leakoff = self.rock.leakoff
        if dpp < -666.0*MPa:
            dpp = self.rock.dPp
        if Vinj < 0:
            Vinj = self.rock.Vinj
        if Qinj < 0:
            Qinj = self.rock.Qinj
            
        #user status update
        print( '+> dynamic stimulation with injection volume (Vinj) = %.0f and rate (Qinj) = %.3f' %(Vinj,Qinj))
        
        #bottom hole pressure, reservoir pore pressure
        bhp = self.rock.BH_P #Pa
        #production well pressure
        pwp = bhp + dpp #Pa
            
        #get index of injection and production wells (index will match p_q and i_q from flow solver outputs)
        i_div = 0
        i_key = []
        p_div = 0
        p_key = []
        for i in range(0,len(self.wells)):
            if (int(self.wells[i].typ) in [typ('injector')]):
                i_key += [int(i)]
                i_div += 1
            if (int(self.wells[i].typ) in [typ('producer')]):
                p_key += [int(i)]
                p_div += 1
        i_key = np.asarray(i_key,dtype=int)
        p_key = np.asarray(p_key,dtype=int)
        
        #initialize targets for stimulation
        #if target == []:
        if not target:
            target = i_key
        else:
            pass

        #looping variables
        vol_ini = 0.0 #m3
        vol_new = 0.0 #m3
        vol_old = 0.0 #m3
        vol_rem = np.ones(i_div,dtype=float)*Vinj #m3
        Pis = [] #Pa
        Qis = [] #m3/s
        completed = np.zeros(i_div,dtype=bool) #T/F segment stim complete
        hydrofrac = np.zeros(i_div,dtype=bool) #T/F hydrofrac instability detection
        stabilize = np.zeros(i_div,dtype=bool) #T/F detection of stable flow
        tip = np.ones(i_div,dtype=float)*bhp #trial injection pressure
        dpi = np.ones(i_div,dtype=float)*-self.rock.dPi #trial pressure incrementer
        
        #if target is specified
        #if target != []:
        if target.any():
            for i in range(0,i_div):
                #match targets to i_key
                if not(i_key[i] in list(target)):
                    completed[i] = True
        
        #initial fracture parameters and network volume
        self.re_init()
        for i in range(0,len(self.faces)):
            self.faces[i].Pmax = bhp
            self.faces[i].Pcen = bhp
            self.GR_bh(i)
            if typ(self.faces[i].typ) != 'boundary':
                vol_ini += (4.0/3.0)*pi*0.25*self.faces[i].dia**2.0*0.5*self.faces[i].bd
        vol_old = vol_ini
        
        #stimulation loop
        if visuals:
            Rs = []
            ts = []
            Ps = []
            ws = []
            Vs = []
            Pn = []
        iters = 0
        maxit = 20
        while 1:
            #loop breaker
            if iters >= maxit:
                print( '-> rock stimulation halted at %i iterations' %(iters))
                break
            iters += 1
            print( '\n[%i] rock stimulation step' %(iters))
            
            #get test injection pressures
            for i in range(0,i_div):
                #set to boundary pressure if interval was completed
                if completed[i]:
                    tip[i] = bhp
                #set with pressure incrementer if not completed
                else:
                    tip[i] = self.rock.s3 + dpi[i]

            #boundary condition placeholders
            q_well = np.full(len(self.wells),None)
            p_well = np.full(len(self.wells),None)
            
            #set injection boundary conditions
            for i in range(0,i_div):
                p_well[i_key[i]] = tip[i]
            
            #set production boundary conditions
            for i in range(0,p_div):
                p_well[p_key[i]] = pwp
                
#            #error tracking
#            print '@@@ hydrofrac & completed & q_well & p_well'
#            print hydrofrac
#            print completed
#            print q_well
#            print p_well

            #solve flow with pressure drive
            self.get_flow(p_bound=bhp,p_well=p_well,q_well=q_well)
            
            #fetch pressure and flow rates #@@@@ this may need updating
            Pis += [tip]
            Qi = []
            for i in range(0,i_div):
                Qi += [self.p_q[i_key[i]]]
            Qis += [Qi]
            #Qis += [np.max(self.p_q)]
            
#            print '@@@ p_p and p_q'
#            print self.p_p
#            print self.p_q
#            
#            print '@@@ tip & Qi'
#            print tip
#            print Qi
            
            #create vtk
            if visuals:
                fname2 = fname + '_A_%i' %(iters)
                self.build_vtk(fname2)
                
            #stimulation complete if pressure driven injection rate exceeds stimulation injection rate in all wells
            #i_q, p_q, b_q are + for flow into the frac network
            for i in range(0,i_div):
                if Qi[i] > Qinj:
                    completed[i] = True
                    stabilize[i] = True
            
            #break if all are compeleted
            if np.sum(completed) == i_div:
                print( '-> stimulation complete: full flow acheived')
                break     
                
            #get max pressure on each fracture from all the nodes associated with that fracture
            face_pmax = np.ones(len(self.faces),dtype=float)*bhp
            for i in range(0,self.pipes.num):
                if (int(self.pipes.typ[i]) in [typ('boundary'),typ('fracture'),typ('propped'),typ('choke')]): #don't confuse well ID with face ID
                    face_pmax[self.pipes.fID[i]] = np.max([face_pmax[self.pipes.fID[i]], self.nodes.p[self.pipes.n0[i]], self.nodes.p[self.pipes.n1[i]]])
            #update fracture properties; stimulate fractures; grow fractures; calculate new fracture volume
            if visuals:
                R = []
                w = []
                V = []
                P = []
            nat_stim = False
            vol_new = 0.0
            num_stim = 0
            for i in range(0,len(self.faces)):
                #record maximum and center node pressures
                self.faces[i].Pmax = face_pmax[i]
                self.faces[i].Pcen = self.nodes.p[self.faces[i].ci]
                #compute fracture properties, if stimulated acknowledge it
                nat_stim += self.GR_bh(i)
                #calculate new fracture volume
                if typ(self.faces[i].typ) != 'boundary':
                    vol_new += (4.0/3.0)*pi*0.25*self.faces[i].dia**2.0*0.5*self.faces[i].bd
                #get maximum number of stimulations
                num_stim = np.max([self.faces[i].stim,num_stim])
                #identify if hydrofrac
                if self.faces[i].hydroprop:
                    #hydrofrac = True
                    for j in range(0,i_div):
                        #only record for intervals that are hydropropped
                        if not(completed[j]):
                            hydrofrac[j] = True
                #variable tracking for visuals
                if visuals:
                    R += [0.5*self.faces[i].dia]
                    w += [self.faces[i].bd]
                    V += [(4.0/3.0)*pi*0.25*self.faces[i].dia**2.0*0.5*self.faces[i].bd]
                    P += [self.faces[i].Pcen]
            if visuals:
                Rs += [R]
                ws += [w]
                Vs += [V]
                Pn += [P]
                
            #remaining injection volume for stimulation accounting for leakoff volume
            if visuals:
                t = []
                P = []
            print( '   - volume remaining')
            for i in range(0,i_div):
                #only modify stimualtions if stage is not yet completed                
                if not(completed[i]):
                    #calculate volume change before next fracture in chain will be triggered
                    time_step = (vol_new-vol_old)/(Qinj-Qi[i])
                    vol_rem[i] = vol_rem[i] - time_step*Qinj
                    print( '   - (%i) volume remaining %.3f m3' %(i,vol_rem[i]))
                    
                    #stage complete if target volume is reached
                    if vol_rem[i] < 0.0:
                        print( '   $ (%i) completed by reaching target injection volume' %(i))
                        completed[i] = True
                        
                    #stimulate hydraulic fractures if criteria met
                    #if (nat_stim == False) and (tip > self.rock.s3+0.1*MPa) and (hydrofrac == False): #and (np.max(self.p_p) < 0.0):
                    elif (hydrofrac[i] == False) and (tip[i] > (self.rock.s3 + self.rock.hfmcc)):
                        print( '   ! (%i) hydraulic fractures' %(i))
                        hydrofrac[i] = True
                        #seed hydraulic fracture
                        self.gen_stimfracs(target=i_key[i],perfs=perfs,
                                      f_dia = [2.0*r_perf,0.0],
                                      f_azn = [self.rock.s3Azn+np.pi/2.0, self.rock.s3AznVar],
                                      f_dip = [np.pi/2.0-self.rock.s3Dip, self.rock.s3DipVar],
                                      clear=False)      
                    #if insufficient pressure and insufficient rate or too many repeated stimulations, increase pressure
                    elif ((nat_stim == False) or (((int(num_stim) + 1) % int(self.rock.stim_limit)) == 0)):
                        dpi[i] += self.rock.dPi
                        print( '   + (%i) pressure increased to %.3f' %(i,self.rock.s3+dpi[i]))
                    if visuals:
                        t += [time_step]
                        P += [tip[i]]
                
            #update fracture network volume
            vol_old = vol_new
            
            #visuals
            if visuals:
                ts += [t]
                Ps += [P]
        
        #@@@
        print( '\n[B] Final flow solve')
        
        #***** reset pressures and fracture geometry
        for i in range(0,len(self.faces)):
            self.faces[i].Pmax = bhp
            self.faces[i].Pcen = bhp
            self.GR_bh(i,fix=True)
        
        #***** final pressure calculation to set facture apertures
        #fixed pressure solve
        q_well = np.full(len(self.wells),None)
        p_well = np.full(len(self.wells),None)
        #set injection boundary conditions using maximum values
        for i in range(0,i_div):
            tip[i] = self.rock.s3 + dpi[i]
            p_well[i_key[i]] = tip[i]
        #set production boundary conditions
        for i in range(0,p_div):
            p_well[p_key[i]] = pwp
            
#        #error tracking
#        print '@@@ hydrofrac & completed & q_well & p_well'
#        print hydrofrac
#        print completed
#        print q_well
#        print p_well
            
        #solve flow
        self.get_flow(p_bound=bhp,p_well=p_well,q_well=q_well,reinit=False)
        #get max pressure on each fracture from all the nodes associated with that fracture
        face_pmax = np.ones(len(self.faces),dtype=float)*bhp
        for i in range(0,self.pipes.num):
            if (int(self.pipes.typ[i]) in [typ('boundary'),typ('fracture'),typ('propped'),typ('choke')]): #don't confuse well ID with face ID
                face_pmax[self.pipes.fID[i]] = np.max([face_pmax[self.pipes.fID[i]], self.nodes.p[self.pipes.n0[i]], self.nodes.p[self.pipes.n1[i]]])
        #update fracture properties without stimualtion
        for i in range(0,len(self.faces)):
            #record maximum and center node pressures
            self.faces[i].Pmax = face_pmax[i]
            self.faces[i].Pcen = self.nodes.p[self.faces[i].ci]
            #compute fracture properties
            self.GR_bh(i,fix=True)
        
#        print '@@@ p_p and p_q'
#        print self.p_p
#        print self.p_q
        
        #***** flow rate solve for heat transfer solution
        q_well = np.full(len(self.wells),None)
        p_well = np.full(len(self.wells),None)
        #set injection boundary conditions using flow values, unless stable flow was never acheived (e.g., hydrofrac only)
        for i in range(0,i_div):
            if stabilize[i]:
                q_well[i_key[i]] = Qinj
            else:
                p_well[i_key[i]] = tip[i]
        #set production boundary conditions
        for i in range(0,p_div):
            p_well[p_key[i]] = pwp
            
            
#        #error tracking
#        print '@@@ stabilize & hydrofrac & completed & q_well & p_well'
#        print stabilize
#        print hydrofrac
#        print completed
#        print q_well
#        print p_well
        
        #solve flow
        self.get_flow(p_bound=bhp,p_well=p_well,q_well=q_well,reinit=False,useprior=True)
        
#        print '@@@ p_p and p_q'
#        print self.p_p
#        print self.p_q
        
        #***** solver overrides for key inputs and outputs
        Pi = []
        Qi = []
        for i in range(0,i_div):
            #locate injection node
            source = self.wells[i_key[i]].c0
            ck, j = self.nodes.add(source)
            if stabilize[i] and hydrofrac[i]:
                self.nodes.p[j] = tip[i]
                Pi += [tip[i]]
            else:
                Pi += [self.nodes.p[j]]
            Qi += [self.p_q[i_key[i]]]
        Pis += [Pi]
        Qis += [Qi]
        
#        print '@@@ Pi & Qi'
#        print Pi
#        print Qi
        
        #final visualization
        Qis = np.asarray(Qis)
        Pis = np.asarray(Pis)
        if visuals:
            #create vtk with final flow data
            fname2 = fname + '_B'
            self.build_vtk(fname2)
#            #create plot of Q and P
#            fig = pylab.figure(figsize=(8.0, 6.0), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
#            ax1 = fig.add_subplot(111)
#            for i in range(0,i_div):
#                ax1.plot(Qis[:,i],Pis[:,i]/MPa,linewidth=0.5,label='%i' %(i))
#            ax1.set_xlabel('Injection Rate (m3/s)')
#            ax1.set_ylabel('Injection Pressure (MPa)')
#            #ax1.legend(loc='upper right', prop={'size':8}, ncol=1, numpoints=1)
            self.v_Rs = Rs
            self.v_ts = ts
            self.v_Ps = Ps
            self.v_ws = ws
            self.v_Vs = Vs
            self.v_Pn = Pn
            
    # ************************************************************************
    # stimulation & flow
    # ************************************************************************
    def stim_and_flow(self,Vstim = -1.0, Qstim = -1.0,
                      Vinj = -1.0, Qinj = -1.0, dpp = -666.6*MPa,
                      sand = -1.0, leakoff = -1.0,
                      target=0, perfs=-1, r_perf=-1.0,
                      clear = True, visuals = True, fname = 'stim'):
        #fetch defaults
        if perfs < 0:
            perfs = self.rock.perf
        if r_perf < 0:
            r_perf = self.rock.r_perf
        if sand < 0:
            sand = self.rock.sand
        if leakoff < 0:
            leakoff = self.rock.leakoff
        if dpp < -666.0*MPa:
            dpp = self.rock.dPp
        if Vinj < 0:
            Vinj = self.rock.Vinj
        if Qinj < 0:
            Qinj = self.rock.Qinj
        if Vstim < 0:
            Vstim = self.rock.Vstim
        if Qstim < 0:
            Qstim = self.rock.Qstim
        
        #Solve stimulation
        self.dyn_stim(Vinj=Vstim,Qinj=Qstim,target=target,
                      visuals=visuals,fname=(fname + '_stim'))
        
        #Solve production
        self.dyn_stim(Vinj=Vinj,Qinj=Qinj,target=target,
                      visuals=visuals,fname=(fname + '_prod'))
        
    # ************************************************************************
    # stimulation & flow
    # ************************************************************************
    def dyn_flow(self,
                      Vinj = -1.0, Qinj = -1.0, dpp = -666.6*MPa,
                      sand = -1.0, leakoff = -1.0,
                      target=0, perfs=-1, r_perf=-1.0,
                      clear = True, visuals = True, fname = 'stim'):
        #fetch defaults
        if perfs < 0:
            perfs = self.rock.perf
        if r_perf < 0:
            r_perf = self.rock.r_perf
        if sand < 0:
            sand = self.rock.sand
        if leakoff < 0:
            leakoff = self.rock.leakoff
        if dpp < -666.0*MPa:
            dpp = self.rock.dPp
        if Vinj < 0:
            Vinj = self.rock.Vinj
        if Qinj < 0:
            Qinj = self.rock.Qinj
        
        #Solve production
        self.dyn_stim(Vinj=Vinj,Qinj=Qinj,target=target,
                      visuals=visuals,fname=(fname + '_prod'))
    
    # ************************************************************************
    # basic validation of fracture geometry
    # ************************************************************************
    def detournay_visc(self,Q0=0.08):
        #plot initialization
        fig = pylab.figure(figsize=(15.0, 8.5), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
        font = {'family' : 'serif',   'size'   : 16}
        pylab.rc('font', serif='Arial')
        pylab.rc('font', **font)
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(234)
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(236)
        
        #normalized constants
        E_prime = self.rock.ResE/(1.0-self.rock.Resv**2.0)
        K_prime = 4.0*self.rock.Kic*(2.0/pi)**0.5
        mu_prime = 12.0*self.rock.Poremu
        
        #time
        time = np.linspace(1000.0,200000.0,100,dtype=float)
        rho = np.linspace(0.001,0.999,100,dtype=float)
        drho = rho[1] - rho[0]
        wst = []
        pst = []
        rst = []
        Vt = []
        Pt = []
        
        for t in time:
            #radius
            sca_radi = 0.0
            sca_Kscas = []
            sca_apers = []
            sca_press = []
            for p in range(0,len(rho)):
                #viscosity dominated if K_scaled < 1.0
                K_scaled = K_prime*(t**2.0/(mu_prime**5.0 * Q0**3.0 * E_prime**13.0))**(1.0/18.0)
                
                #scaled aperture (ohmega_bar_m0) and scaled pressure (pi_m0)
                scaled_aper = (((0.6846*70.0**0.5)/3.0  + (13.0*rho[p]-6.0)*(0.07098*4.0*5.0**0.5)/9.0)*(1.0 - rho[p])**(2.0/3.0) + 
                               0.09269*((1.0-rho[p])**0.5*8.0/pi - np.arccos(rho[p])*rho[p]*8.0/pi))
                scaled_pres = 0.3581*(2.479 - 2.0/(3.0*(1.0-rho[p])**(1.0/3.0))) - 0.09269*(np.log(rho[p]/2.0)+1.0)
                
                sca_Kscas += [K_scaled]
                sca_apers += [scaled_aper]
                sca_press += [scaled_pres]
                
                #scaled radius (gamma_m0)
                if p != 0:
                    sca_radi += 0.5*(scaled_aper + sca_apers[p-1])*0.5*(rho[p] + rho[p-1])*drho
            sca_radi = (2.0*pi*sca_radi)**(-1.0/3.0)

            #convert to arrays for math
            sca_Kscas = np.asarray(sca_Kscas)
            sca_apers = np.asarray(sca_apers)
            sca_press = np.asarray(sca_press)
            
            #less scaled aperture (ohmega_m0)
            n_apers = sca_apers*sca_radi
            
            #lets now undo this irritating scaling crap
            Lm = (E_prime*Q0**3.0*t**4.0/mu_prime)**(1.0/9.0)
            em = (mu_prime/(E_prime*t))**(1.0/3.0)
            ws = em*Lm*n_apers
            ps = em*E_prime*sca_press
            rs = Lm*rho*sca_radi
            dr = Lm*drho*sca_radi
            
            #store for plotting
            wst += [ws]
            pst += [ps]
            rst += [rs]
            
            #now for what we actually care about: volume and radius
            vol = 0.0
            pnet = 0.0
            for i in range(1,len(rs)):
                vol += 2.0*pi*(0.5*(rs[i-1]+rs[i]))*(0.5*(ws[i-1]+ws[i]))*dr
                pnet += 2.0*pi*(0.5*(rs[i-1]+rs[i]))*(0.5*(ps[i-1]+ps[i]))*dr
            pnet = pnet/(pi*rs[-1]**2.0)
            Vt += [vol]
            Pt += [pnet]
            
            #plot
            ax1.plot(rs,ps)
            ax2.plot(rs,ws)
        #ax3.plot(time,np.asarray(Vt))
        #ax4.plot(time,np.asarray(Pt))
        Vt = np.asarray(Vt)
        Pt = np.asarray(Pt)
        rst = np.asarray(rst)
        ax3.plot(rst[:,-1],Vt)
        ax4.plot(rst[:,-1],Pt)
        ax5.plot(rst[:,-1],Vt/(pi*rst[:,-1]**2.0))
        
        #labels
        ax1.set_xlabel('Radial Distance (m)')
        ax1.set_ylabel('Pressure (Pa)')
        
        ax2.set_xlabel('Radial Distance (m)')
        ax2.set_ylabel('Aperture (m)')
        
        ax3.set_xlabel('Radius (m)')
        ax3.set_ylabel('Volume (m3)')
        
        ax4.set_xlabel('Radius (m)')
        ax4.set_ylabel('Avg. Net Pressure (Pa)')
        
        ax5.set_xlabel('Radius (m)')
        ax5.set_ylabel('Avg. Aperture (m)')
        
#        print 'det_Vs'
#        print Vt
#        print 'det_Rs'
#        print rst[:,-1]
#        print 'det_ws'
#        print Vt/(pi*rst[:,-1]**2.0)
#        print 'det_ts'
#        print time
#        print 'det_Pnet'
#        print Pt
#        print 'det_Pinj'
#        print np.asarray(pst)[:,0]


# ****************************************************************************
#### validation: heat flow one pipe
# ****************************************************************************
if False: #validation: heat flow one pipe
    #controlled model
    geom = []
    geom = mesh()
    
    #***** input block settings *****
    #rock properties
    geom.rock.size = 200.0 #m
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
    geom.rock.alpha = np.asarray([-0.028/MPa,-0.028/MPa,-0.028/MPa])
    geom.rock.gamma = np.asarray([0.01,0.01,0.01])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.05,0.05,0.05])
    geom.rock.b = np.asarray([0.8,0.8,0.8])
    geom.rock.N = np.asarray([0.2,0.5,1.2])
    geom.rock.bh = np.asarray([0.00005,0.0001,0.003])
    geom.rock.bh_min = 0.0030001 #0.00005 #m #!!!
    geom.rock.bh_max = 0.003 #0.01 #m #!!!
    geom.rock.bh_bound = 0.001
    geom.rock.f_roughness = np.random.uniform(0.7,1.0) #0.8
    #well parameters
    geom.rock.w_count = 1 #2 #wells
    geom.rock.w_spacing = 50.0 #m
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
    geom.rock.H_ConvCoef = 3.0 #kW/m2-K
    geom.rock.dT0 = 10.0 #K
    geom.rock.dE0 = 500.0 #kJ/m2
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
    geom.rock.r_perf = 80.0 #m
    geom.rock.sand = 0.3 #sand ratio in frac fluid
    geom.rock.leakoff = 0.0 #Carter leakoff
    geom.rock.dPp = -2.0*MPa #production well pressure drawdown
    geom.rock.dPi = 0.1*MPa 
    geom.rock.stim_limit = 5
    geom.rock.Qinj = 0.01 #m3/s
    geom.rock.Qstim = 0.08 #m3/s
    geom.rock.Vstim = 150.0 #m3
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
    geom.rock.phi = np.asarray([35.0*deg,35.0*deg,35.0*deg]) #rad
    geom.rock.mcc = np.asarray([10.0,10.0,10.0])*MPa #Pa
    geom.rock.hfmcc = 0.0*MPa #0.1*MPa
    geom.rock.hfphi = 30.0*deg #30.0*deg
    geom.Kic = 1.5*MPa #Pa-m**0.5
    #**********************************

    #recalculate base parameters
    geom.rock.re_init()
    
    #generate domain
    geom.gen_domain()

#    #generate natural fractures
#    geom.gen_joint_sets()
    
#    #generate wells
#    wells = []
#    geom.gen_wells(True,wells)
    
    #reference fracture for properties
    c0 = [0.0,0.0,0.0]
    dia = 100.0
    strike = 90.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(True,c0,dia,strike,dip)
    geom.fracs[-1].bh = 0.003
    
    #reference wells
    geom.wells = [line(0.0,-25.0,0.0,1.0,0.0*deg,0.0*deg,'injector',geom.rock.ra,geom.rock.rgh),
                  line(50.0,-25.0,0.0,1.0,0.0*deg,0.0*deg,'producer',geom.rock.ra,geom.rock.rgh)]
    
    geom.re_init()
    
    #custom injector
    geom.add_flowpath(np.asarray([0.0,-25.0,0.0]),
                     np.asarray([0.0,0.0,0.0]),
                     25.0,
                     1.0,
                     typ('injector'),
                     0)
    geom.add_flowpath(np.asarray([50.0,-25.0,0.0]),
                     np.asarray([50.0,0.0,0.0]),
                     25.0,
                     1.0,
                     typ('producer'),
                     1)
    geom.add_flowpath(np.asarray([0.0,0.0,0.0]),
                     np.asarray([50.0,0.0,0.0]),
                     50.0,
                     50.0,
                     typ('fracture'),
                     6)  
    
    #random identifier (statistically should be unique)
    pin = np.random.randint(100000000,999999999,1)[0]
    
#    #stimulate
#    geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
#                  visuals=False,fname='stim')
    
#    #flow
#    geom.dyn_stim(Vinj=geom.rock.Vinj,Qinj=geom.rock.Qinj,target=[],
#                  visuals=False,fname='run_%i' %(pin))
    
    geom.get_flow(p_bound=geom.rock.BH_P,p_well=np.asarray([None,geom.rock.BH_P]),q_well=np.asarray([geom.rock.Qinj*0.008,None]),reinit=False,useprior=False)
    
    #heat flow
    geom.get_heat(plot=True)
    plt.savefig('plt_%i.png' %(pin), format='png')
    plt.close()
    
    #show flow model
    geom.build_vtk(fname='fin_%i' %(pin))
        
#    if True: #3D temperature visual
#        geom.build_pts(spacing=5.0,fname='fin_%i' %(pin))
#        
#    #save primary inputs and outputs
#    x = geom.save('inputs_results_valid.txt',pin)
    
#    #stereoplot
#    geom.rock.stress.plot_Pc(geom.rock.phi[1], geom.rock.mcc[1], filename='Pc_stereoplot.png')
    
#    #check vs Detournay
#    geom.detournay_visc(geom.rock.Qstim)
#    
#    #fracture growth plot
#    Vs = np.asarray(list(geom.v_Vs[3:]))[:,-2:]
#    Rs = np.asarray(list(geom.v_Rs[3:]))[:,-2:]
#    ws = np.asarray(list(geom.v_ws[3:]))[:,-2:]
#    ts = np.asarray(list(geom.v_ts[3:]))
#    Ps = np.asarray(list(geom.v_Ps[3:]))
#    Pn = np.asarray(list(geom.v_Pn[3:]))[:,-2:]

    #show plots
    pylab.show()

# ****************************************************************************
#### validation: heat flow
# ****************************************************************************
if False: #validation: heat flow
    #controlled model
    geom = []
    geom = mesh()
    
    #***** input block settings *****
    #rock properties
    geom.rock.size = 100.0 #m
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
    geom.rock.alpha = np.asarray([-0.028/MPa,-0.028/MPa,-0.028/MPa])
    geom.rock.gamma = np.asarray([0.01,0.01,0.01])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.05,0.05,0.05])
    geom.rock.b = np.asarray([0.8,0.8,0.8])
    geom.rock.N = np.asarray([0.2,0.5,1.2])
    geom.rock.bh = np.asarray([0.00005,0.0001,0.003])
    geom.rock.bh_min = 0.0030001 #0.00005 #m #!!!
    geom.rock.bh_max = 0.003 #0.01 #m #!!!
    geom.rock.bh_bound = 0.001
    geom.rock.f_roughness = np.random.uniform(0.7,1.0) #0.8
    #well parameters
    geom.rock.w_count = 1 #2 #wells
    geom.rock.w_spacing = 50.0 #m
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
    geom.rock.H_ConvCoef = 3.0 #kW/m2-K
    geom.rock.dT0 = 10.0 #K
    geom.rock.dE0 = 500.0 #kJ/m2
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
    geom.rock.r_perf = 80.0 #m
    geom.rock.sand = 0.3 #sand ratio in frac fluid
    geom.rock.leakoff = 0.0 #Carter leakoff
    geom.rock.dPp = -2.0*MPa #production well pressure drawdown
    geom.rock.dPi = 0.1*MPa 
    geom.rock.stim_limit = 5
    geom.rock.Qinj = 0.01 #m3/s
    geom.rock.Qstim = 0.08 #m3/s
    geom.rock.Vstim = 150.0 #m3
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
    geom.rock.phi = np.asarray([35.0*deg,35.0*deg,35.0*deg]) #rad
    geom.rock.mcc = np.asarray([10.0,10.0,10.0])*MPa #Pa
    geom.rock.hfmcc = 0.0*MPa #0.1*MPa
    geom.rock.hfphi = 30.0*deg #30.0*deg
    geom.Kic = 1.5*MPa #Pa-m**0.5
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
    
    #random identifier (statistically should be unique)
    pin = np.random.randint(100000000,999999999,1)[0]
    
    #stimulate
    geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
                  visuals=False,fname='stim')
    
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
        geom.build_pts(spacing=2.5,fname='fin_%i' %(pin))
        
    #save primary inputs and outputs
    x = geom.save('inputs_results_valid.txt',pin)
    
#    #stereoplot
#    geom.rock.stress.plot_Pc(geom.rock.phi[1], geom.rock.mcc[1], filename='Pc_stereoplot.png')
    
#    #check vs Detournay
#    geom.detournay_visc(geom.rock.Qstim)
#    
#    #fracture growth plot
#    Vs = np.asarray(list(geom.v_Vs[3:]))[:,-2:]
#    Rs = np.asarray(list(geom.v_Rs[3:]))[:,-2:]
#    ws = np.asarray(list(geom.v_ws[3:]))[:,-2:]
#    ts = np.asarray(list(geom.v_ts[3:]))
#    Ps = np.asarray(list(geom.v_Ps[3:]))
#    Pn = np.asarray(list(geom.v_Pn[3:]))[:,-2:]

    #show plots
    pylab.show()

# ****************************************************************************
#### validation: stress stereonet
# ****************************************************************************        
if False: #validation: stress stereonet
    #controlled model
    geom = []
    geom = mesh()
    
    #***** input block settings *****
    #rock properties
    geom.rock.size = 1000.0 #m
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
    geom.rock.alpha = np.asarray([-0.028/MPa,-0.028/MPa,-0.028/MPa])
    geom.rock.gamma = np.asarray([0.01,0.01,0.01])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.05,0.05,0.05])
    geom.rock.b = np.asarray([0.8,0.8,0.8])
    geom.rock.N = np.asarray([0.2,0.5,1.2])
    geom.rock.bh = np.asarray([0.00005,0.0001,0.003])
    geom.rock.bh_min = 0.0030001 #0.00005 #m #!!!
    geom.rock.bh_max = 0.003 #0.01 #m #!!!
    geom.rock.bh_bound = 0.001
    geom.rock.f_roughness = np.random.uniform(0.7,1.0) #0.8
    #well parameters
    geom.rock.w_count = 0 #2 #wells
    geom.rock.w_spacing = 400.0 #m
    geom.rock.w_length = 500.0 #800.0 #m
    geom.rock.w_azimuth = 0.0*deg #rad
    geom.rock.w_dip = 0.0*deg #rad
    geom.rock.w_proportion = 0.5 #m/m
    geom.rock.w_phase = 90.0*deg #rad
    geom.rock.w_toe = 0.0*deg #rad
    geom.rock.w_skew = 0.0*deg #rad
    geom.rock.w_intervals = 2 #breaks in well length
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
    geom.rock.H_ConvCoef = 3.0 #kW/m2-K
    geom.rock.dT0 = 10.0 #K
    geom.rock.dE0 = 500.0 #kJ/m2
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
    geom.rock.r_perf = 50.0 #m
    geom.rock.sand = 0.3 #sand ratio in frac fluid
    geom.rock.leakoff = 0.0 #Carter leakoff
    geom.rock.dPp = -2.0*MPa #production well pressure drawdown
    geom.rock.dPi = 0.1*MPa 
    geom.rock.stim_limit = 5
    geom.rock.Qinj = 0.003 #m3/s
    geom.rock.Qstim = 0.08 #m3/s
    geom.rock.Vstim = 150.0 #m3
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
    geom.rock.phi = np.asarray([35.0*deg,35.0*deg,35.0*deg]) #rad
    geom.rock.mcc = np.asarray([10.0,10.0,10.0])*MPa #Pa
    geom.rock.hfmcc = 0.0*MPa #0.1*MPa
    geom.rock.hfphi = 30.0*deg #30.0*deg
    geom.Kic = 1.5*MPa #Pa-m**0.5
    #**********************************

    #recalculate base parameters
    geom.rock.re_init()
    
    #generate domain
    geom.gen_domain()

    #generate natural fractures
    geom.gen_joint_sets()
    
    #horizontal
    c0 = [0.0,0.0,-500.0]
    dia = 150.0
    strike = 0.0*deg
    dip = 0.0*deg
    geom.gen_fixfrac(True,c0,dia,strike,dip)
    
    #verical x-dir
    c0 = [0.0,500.0,0.0]
    dia = 150.0
    strike = 90.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    #slope x-dir
    c0 = [0.0,250.0,-250.0]
    dia = 150.0
    strike = 90.0*deg
    dip = 45.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    
    #vertical y-dir
    c0 = [-500.0,0.0,0.0]
    dia = 150.0
    strike = 0.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    #slope y-dir
    c0 = [-250.0,0.0,-250.0]
    dia = 150.0
    strike = 0.0*deg
    dip = 45.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    
    #skew vertical
    c0 = [-353.0,353.0,0.0]
    dia = 150.0
    strike = 45.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    #skew slope
    c0 = [-176.0,176.0,-250.0]
    dia = 150.0
    strike = 45.0*deg
    dip = 45.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    #******

    #verical x-dir
    c0 = [0.0,-500.0,0.0]
    dia = 150.0
    strike = 270.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    #slope x-dir
    c0 = [0.0,-250.0,-250.0]
    dia = 150.0
    strike = 270.0*deg
    dip = 45.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    
    #vertical y-dir
    c0 = [500.0,0.0,0.0]
    dia = 150.0
    strike = 180.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    #slope y-dir
    c0 = [250.0,0.0,-250.0]
    dia = 150.0
    strike = 180.0*deg
    dip = 45.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    
    #skew vertical
    c0 = [353.0,-353.0,0.0]
    dia = 150.0
    strike = 225.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    #skew slope
    c0 = [176.0,-176.0,-250.0]
    dia = 150.0
    strike = 225.0*deg
    dip = 45.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    #******

    #skew vertical
    c0 = [-353.0,-353.0,0.0]
    dia = 150.0
    strike = 315.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    #skew slope
    c0 = [-176.0,-176.0,-250.0]
    dia = 150.0
    strike = 315.0*deg
    dip = 45.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    #******

    #skew vertical
    c0 = [353.0,353.0,0.0]
    dia = 150.0
    strike = 135.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    #skew slope
    c0 = [176.0,176.0,-250.0]
    dia = 150.0
    strike = 135.0*deg
    dip = 45.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    #******
    
    #generate wells
    wells = []
    geom.gen_wells(True,wells)
    
    #random identifier (statistically should be unique)
    pin = np.random.randint(100000000,999999999,1)[0]
    
    #stimulate
    geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
                  visuals=False,fname='stim')
    
#    #flow
#    geom.dyn_stim(Vinj=geom.rock.Vinj,Qinj=geom.rock.Qinj,target=[],
#                  visuals=False,fname='run_%i' %(pin))
#    
#    #heat flow
#    geom.get_heat(plot=True)
#    plt.savefig('plt_%i.png' %(pin), format='png')
#    plt.close()
#    
    #show flow model
    geom.build_vtk(fname='fin_%i' %(pin))
#        
#    if True: #3D temperature visual
#        geom.build_pts(spacing=50.0,fname='fin_%i' %(pin))
        
    #save primary inputs and outputs
    x = geom.save('inputs_results_valid.txt',pin)
    
    #stereoplot
    geom.rock.stress.plot_Pc(geom.rock.phi[1], geom.rock.mcc[1], filename='Pc_stereoplot.png')
    
#    #check vs Detournay
#    geom.detournay_visc(geom.rock.Qstim)
#    
#    #fracture growth plot
#    Vs = np.asarray(list(geom.v_Vs[3:]))[:,-2:]
#    Rs = np.asarray(list(geom.v_Rs[3:]))[:,-2:]
#    ws = np.asarray(list(geom.v_ws[3:]))[:,-2:]
#    ts = np.asarray(list(geom.v_ts[3:]))
#    Ps = np.asarray(list(geom.v_Ps[3:]))
#    Pn = np.asarray(list(geom.v_Pn[3:]))[:,-2:]

    #show plots
    pylab.show()
        
        
# ****************************************************************************
#### validation: inj only with hydrofrac
# ****************************************************************************           
if False: #validation: inj only with hydrofrac
    #controlled model
    geom = []
    geom = mesh()
    
    #***** input block settings *****
    #rock properties
    geom.rock.size = 1000.0 #m
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
    geom.rock.alpha = np.asarray([-0.028/MPa,-0.028/MPa,-0.028/MPa])
    geom.rock.gamma = np.asarray([0.01,0.01,0.01])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.05,0.05,0.05])
    geom.rock.b = np.asarray([0.8,0.8,0.8])
    geom.rock.N = np.asarray([0.2,0.5,1.2])
    geom.rock.bh = np.asarray([0.00005,0.0001,0.003])
    geom.rock.bh_min = 0.0030001 #0.00005 #m #!!!
    geom.rock.bh_max = 0.003 #0.01 #m #!!!
    geom.rock.bh_bound = 0.001
    geom.rock.f_roughness = np.random.uniform(0.7,1.0) #0.8
    #well parameters
    geom.rock.w_count = 0 #2 #wells
    geom.rock.w_spacing = 400.0 #m
    geom.rock.w_length = 500.0 #800.0 #m
    geom.rock.w_azimuth = 0.0*deg #rad
    geom.rock.w_dip = 0.0*deg #rad
    geom.rock.w_proportion = 0.5 #m/m
    geom.rock.w_phase = 90.0*deg #rad
    geom.rock.w_toe = 0.0*deg #rad
    geom.rock.w_skew = 0.0*deg #rad
    geom.rock.w_intervals = 2 #breaks in well length
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
    geom.rock.H_ConvCoef = 3.0 #kW/m2-K
    geom.rock.dT0 = 10.0 #K
    geom.rock.dE0 = 500.0 #kJ/m2
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
    geom.rock.r_perf = 50.0 #m
    geom.rock.sand = 0.3 #sand ratio in frac fluid
    geom.rock.leakoff = 0.0 #Carter leakoff
    geom.rock.dPp = -2.0*MPa #production well pressure drawdown
    geom.rock.dPi = 0.1*MPa 
    geom.rock.stim_limit = 5
    geom.rock.Qinj = 0.003 #m3/s
    geom.rock.Qstim = 0.08 #m3/s
    geom.rock.Vstim = 100000.0 #m3
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
    geom.rock.phi = np.asarray([35.0*deg,35.0*deg,35.0*deg]) #rad
    geom.rock.mcc = np.asarray([10.0,10.0,10.0])*MPa #Pa
    geom.rock.hfmcc = 0.0*MPa #0.1*MPa
    geom.rock.hfphi = 30.0*deg #30.0*deg
    geom.Kic = 1.5*MPa #Pa-m**0.5
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
    
    #random identifier (statistically should be unique)
    pin = np.random.randint(100000000,999999999,1)[0]
    
    #stimulate
    geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
                  visuals=False,fname='stim')
    
#    #flow
#    geom.dyn_stim(Vinj=geom.rock.Vinj,Qinj=geom.rock.Qinj,target=[],
#                  visuals=False,fname='run_%i' %(pin))
#    
#    #heat flow
#    geom.get_heat(plot=True)
#    plt.savefig('plt_%i.png' %(pin), format='png')
#    plt.close()
#    
#    #show flow model
#    geom.build_vtk(fname='fin_%i' %(pin))
#        
#    if True: #3D temperature visual
#        geom.build_pts(spacing=50.0,fname='fin_%i' %(pin))
        
    #save primary inputs and outputs
    x = geom.save('inputs_results_valid.txt',pin)
    
    #stereoplot
    geom.rock.stress.plot_Pc(geom.rock.phi[1], geom.rock.mcc[1], filename='Pc_stereoplot.png')
    
    #check vs Detournay
    geom.detournay_visc(geom.rock.Qstim)
    
    #fracture growth plot
    Vs = np.asarray(list(geom.v_Vs[3:]))[:,-2:]
    Rs = np.asarray(list(geom.v_Rs[3:]))[:,-2:]
    ws = np.asarray(list(geom.v_ws[3:]))[:,-2:]
    ts = np.asarray(list(geom.v_ts[3:]))
    Ps = np.asarray(list(geom.v_Ps[3:]))
    Pn = np.asarray(list(geom.v_Pn[3:]))[:,-2:]

    #show plots
    pylab.show()

# ****************************************************************************
#### validation: two inj two pro two natfrac
# **************************************************************************** 
if True: #validation: two inj two pro two natfrac 
    #controlled model
    geom = []
    geom = mesh()
    
    #***** input block settings *****
    #rock properties
    geom.rock.size = 1000.0 #m
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
    geom.rock.alpha = np.asarray([-0.028/MPa,-0.028/MPa,-0.028/MPa])
    geom.rock.gamma = np.asarray([0.01,0.01,0.01])
    geom.rock.n1 = np.asarray([1.0,1.0,1.0])
    geom.rock.a = np.asarray([0.05,0.05,0.05])
    geom.rock.b = np.asarray([0.8,0.8,0.8])
    geom.rock.N = np.asarray([0.2,0.5,1.2])
    geom.rock.bh = np.asarray([0.00005,0.0001,0.003])
    geom.rock.bh_min = 0.0030001 #0.00005 #m #!!!
    geom.rock.bh_max = 0.003 #0.01 #m #!!!
    geom.rock.bh_bound = 0.001
    geom.rock.f_roughness = np.random.uniform(0.7,1.0) #0.8
    #well parameters
    geom.rock.w_count = 2 #2 #wells
    geom.rock.w_spacing = 200.0 #m
    geom.rock.w_length = 500.0 #800.0 #m
    geom.rock.w_azimuth = 10.0*deg #rad
    geom.rock.w_dip = 5.0*deg #rad
    geom.rock.w_proportion = 0.5 #m/m
    geom.rock.w_phase = -30.0*deg #rad
    geom.rock.w_toe = -10.0*deg #rad
    geom.rock.w_skew = 10.0*deg #rad
    geom.rock.w_intervals = 2 #breaks in well length
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
    geom.rock.H_ConvCoef = 3.0 #kW/m2-K
    geom.rock.dT0 = 10.0 #K
    geom.rock.dE0 = 500.0 #kJ/m2
    #water base parameters
    geom.rock.PoreRho = 980.0 #kg/m3 starting guess
    geom.rock.Poremu = 0.9*cP #Pa-s
    geom.rock.Porek = 0.01*mD #m2
    geom.rock.Frack = 100.0*mD #m2
    #stimulation parameters
    if geom.rock.w_intervals == 1:
        geom.rock.perf = int(np.random.uniform(1,1))
    else:
        geom.rock.perf = 1
    geom.rock.r_perf = 50.0 #m
    geom.rock.sand = 0.3 #sand ratio in frac fluid
    geom.rock.leakoff = 0.0 #Carter leakoff
    geom.rock.dPp = -2.0*MPa #production well pressure drawdown
    geom.rock.dPi = 0.5*MPa 
    geom.rock.stim_limit = 5
    geom.rock.Qinj = 0.003 #m3/s
    geom.rock.Qstim = 0.08 #m3/s
    geom.rock.Vstim = 100000.0 #m3
    geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
    geom.rock.phi = np.asarray([35.0*deg,35.0*deg,35.0*deg]) #rad
    geom.rock.mcc = np.asarray([10.0,10.0,10.0])*MPa #Pa
    geom.rock.hfmcc = 0.1*MPa #0.1*MPa
    geom.rock.hfphi = 30.0*deg #30.0*deg
    #**********************************

    #recalculate base parameters
    geom.rock.re_init()
    
    #generate domain
    geom.gen_domain()

    #generate natural fractures
    geom.gen_joint_sets()
    
    #generate fixed joint sets
    c0 = [-50.0,80.0,25.0]
    dia = 500.0
    strike = 100.0*deg
    dip = 80.0*deg
    geom.gen_fixfrac(True,c0,dia,strike,dip)
    
    c0 = [50.0,-80.0,-25.0]
    dia = 600.0
    strike = 280.0*deg
    dip = 80.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    c0 = [0.0,0.0,-999.0]
    dia = 999.0
    strike = 0.0*deg
    dip = 0.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    c0 = [0.0,-999.0,0.0]
    dia = 999.0
    strike = 90.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)

    c0 = [-999.0,0.0,0.0]
    dia = 999.0
    strike = 0.0*deg
    dip = 90.0*deg
    geom.gen_fixfrac(False,c0,dia,strike,dip)
    
    #generate wells
    wells = []
    geom.gen_wells(True,wells)
    
    #random identifier (statistically should be unique)
    pin = np.random.randint(100000000,999999999,1)[0]
    
#    #stimulate
#    geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
#                  visuals=False,fname='stim')
    
    #flow
    geom.dyn_stim(Vinj=geom.rock.Vinj,Qinj=geom.rock.Qinj,target=[],
                  visuals=False,fname='run_%i' %(pin))
    
    #heat flow
    geom.get_heat(plot=True)
    plt.savefig('plt_%i.png' %(pin), format='png')
    plt.close()
    
    #show flow model
    geom.build_vtk(fname='fin_%i' %(pin))
        
    if False: #3D temperature visual
        geom.build_pts(spacing=50.0,fname='fin_%i' %(pin))
        
    #save primary inputs and outputs
    x = geom.save('inputs_results_valid.txt',pin)
    
    #show plots
    pylab.show()

# ****************************************************************************
#### randomized inputs w/ scaled flow
# ****************************************************************************         
if False: #randomized inputs w/ scaled flow
    #full randomizer
    for i in range(0,1):
        geom = []
        geom = mesh()
        
#        #***** input block randomizer *****
#        #rock properties
        geom.rock.size = 1500.0 #m
        geom.rock.ResDepth = np.random.uniform(4000.0,8000.0) #6000.0 # m #!!!
        geom.rock.ResGradient = np.random.uniform(40.0,50.0) #50.0 #56.70 # C/km; average = 25 C/km
        geom.rock.ResRho = np.random.uniform(2700.0,2700.0) #2700.0 # kg/m3
        geom.rock.ResKt = np.random.uniform(2.1,2.8) #2.5 # W/m-K
        geom.rock.ResSv = np.random.uniform(1900.0,2200.0) #2063.0 # kJ/m3-K
        geom.rock.AmbTempC = np.random.uniform(20.0,20.0) #25.0 # C
        geom.rock.AmbPres = 0.101 #Example: 0.01 MPa #Atmospheric: 0.101 # MPa
        geom.rock.ResE = np.random.uniform(30.0,90.0)*GPa #50.0*GPa
        geom.rock.Resv = np.random.uniform(0.15,0.35) #0.3
        geom.rock.Ks3 = np.random.uniform(0.3,0.9) #0.5 #!!!
        geom.rock.Ks2 = geom.rock.Ks3 + np.random.uniform(0.0,0.6) # 0.75 #!!!
        geom.rock.s3Azn = 0.0*deg
        geom.rock.s3AznVar = 0.0*deg
        geom.rock.s3Dip = 0.0*deg
        geom.rock.s3DipVar = 0.0*deg
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
                                int(np.random.uniform(0,120)),
                                int(np.random.uniform(0,30))],dtype=int) #count
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
#        r = np.random.exponential(scale=0.25,size=2)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        r = r*(0.100/MPa-0.001/MPa)+0.001/MPa   
#        u1 = -np.min(r)
#        u2 = -np.max(r)
#        u3 = 0.5*(u1+u2)
#        geom.rock.alpha = np.asarray([u1,u3,u2])
        geom.rock.alpha = np.asarray([-0.028/MPa,-0.028/MPa,-0.028/MPa])
        
#        r = np.random.exponential(scale=0.25,size=2)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        r = r*(0.1-0.001)+0.001   
#        u1 = np.min(r)
#        u2 = np.max(r)
#        u3 = 0.5*(u1+u2)
#        geom.rock.gamma = np.asarray([u1,u3,u2])
        geom.rock.gamma = np.asarray([0.01,0.01,0.01])
        
        geom.rock.n1 = np.asarray([1.0,1.0,1.0])
        
#        r = np.random.exponential(scale=0.25,size=2)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        r = r*(0.2-0.012)+0.012   
#        u1 = np.min(r)
#        u2 = np.max(r)
#        u3 = 0.5*(u1+u2)
#        geom.rock.a = np.asarray([u1,u3,u2])
        geom.rock.a = np.asarray([0.05,0.05,0.05])
        
#        u1 = np.random.uniform(0.7,0.9)
#        u2 = np.random.uniform(0.7,0.9)
#        u3 = 0.5*(u1+u2)
#        geom.rock.b = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
        geom.rock.b = np.asarray([0.8,0.8,0.8])
        
#        u1 = np.random.uniform(0.2,1.2)
#        u2 = np.random.uniform(0.2,1.2)
#        u3 = 0.5*(u1+u2)
#        geom.rock.N = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
        geom.rock.N = np.asarray([0.2,0.5,1.2])
        
#        u1 = np.random.uniform(0.00005,0.00015)
#        u2 = np.random.uniform(0.00005,0.00015)
#        u3 = 0.5*(u1+u2)
#        geom.rock.bh = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
        geom.rock.bh = np.asarray([0.00005,0.0001,0.003])
        
        geom.rock.bh_min = 0.00005 #m
        geom.rock.bh_max = 0.01 #0.02000 #m
#        geom.rock.bh_bound = np.random.uniform(0.001,0.005)
        geom.rock.bh_bound = 0.001
        geom.rock.f_roughness = np.random.uniform(0.7,1.0) #0.8
        #well parameters
#        geom.rock.w_count = int(np.random.uniform(1,4)) #2 #wells
#        geom.rock.w_spacing = np.random.uniform(100.0,800.0) #300.0 #m
#        geom.rock.w_length = np.random.uniform(800.0,1600.0) #800.0 #m
#        geom.rock.w_azimuth = geom.rock.s3Azn + np.random.uniform(-10.0,10.0)*deg #rad
#        geom.rock.w_dip = geom.rock.s3Dip + np.random.uniform(-10.0,10.0)*deg #rad
#        geom.rock.w_proportion = np.random.uniform(1.2,0.5) #0.8 #m/m
#        geom.rock.w_phase = np.random.uniform(0.0,360.0)*deg #-90.0*deg #rad
#        geom.rock.w_toe = np.random.uniform(0.0,0.0)*deg #rad
#        geom.rock.w_skew = np.random.uniform(-15.0,15.0)*deg #rad
#        geom.rock.w_intervals = int(np.random.uniform(2,3)) #3 #breaks in well length
#        geom.rock.ra = 0.0254*np.random.uniform(2.0,7.0) #3.0 #0.0254*3.0 #m
        geom.rock.w_count = 2 #2 #wells
        geom.rock.w_spacing = np.random.uniform(100.0,800.0) #300.0 #m
        geom.rock.w_length = 1500.0 #800.0 #m
        geom.rock.w_azimuth = geom.rock.s3Azn + np.random.uniform(-15.0,15.0)*deg #rad
        geom.rock.w_dip = geom.rock.s3Dip + np.random.uniform(-15.0,15.0)*deg #rad
        geom.rock.w_proportion = 0.8 #m/m
        geom.rock.w_phase = -90.0*deg #rad
        geom.rock.w_toe = 0.0*deg #rad
        geom.rock.w_skew = 0.0*deg #rad
        geom.rock.w_intervals = int(np.random.uniform(1,3)) #3 #breaks in well length
        geom.rock.ra = 0.0254*np.random.uniform(3.0,8.0) #3.0 #0.0254*3.0 #m
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
        geom.rock.Poremu = 0.9*cP #Pa-s
        geom.rock.Porek = 0.1*mD #m2
        geom.rock.Frack = 100.0*mD #m2
        #stimulation parameters
        if geom.rock.w_intervals == 1:
            geom.rock.perf = int(np.random.uniform(1,1))
        else:
            geom.rock.perf = 1
        geom.rock.r_perf = 50.0 #m
        geom.rock.sand = 0.3 #sand ratio in frac fluid
        geom.rock.leakoff = 0.0 #Carter leakoff
#        geom.rock.dPp = -1.0*np.random.uniform(1.0,10.0)*MPa #-2.0*MPa #production well pressure drawdown
        geom.rock.dPp = -2.0*MPa #production well pressure drawdown
        geom.rock.dPi = 0.1*MPa #!!!
        geom.rock.stim_limit = 5
    #    geom.rock.Qinj = 0.01 #m3/s
        geom.rock.Qstim = 0.08 #m3/s
        geom.rock.Vstim = 100000.0 #m3
        geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
#        u1 = np.random.uniform(20.0,55.0)*deg
#        u2 = np.random.uniform(20.0,55.0)*deg
#        u3 = 0.5*(u1+u2)
#        geom.rock.phi = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])]) #rad
        geom.rock.phi = np.asarray([25.0*deg,35.0*deg,45.0*deg]) #rad
#        u1 = np.random.uniform(5.0,20.0)*MPa
#        u2 = np.random.uniform(5.0,20.0)*MPa
#        u3 = 0.5*(u1+u2)
#        geom.rock.mcc = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])]) #Pa
        geom.rock.mcc = np.asarray([5.0,10.0,15.0])*MPa #Pa
        geom.rock.hfmcc = np.random.uniform(0.0,0.2)*MPa #0.1*MPa
        geom.rock.hfphi = np.random.uniform(15.0,45.0)*deg #30.0*deg
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
        
        #stimulate
        geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
                      visuals=False,fname='stim')
        
        #test multiple randomly selected flow rates
        #rates = np.random.uniform(0.005,0.08,10)
        rates = [0.01]
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
            x = base.save('inputs_results_valid.txt',pin)
    
    
    #show plots
    pylab.show()
    
# ****************************************************************************
#### EGS collab example randomized inputs w/ scaled flow
# ****************************************************************************  
if False: #EGS collab example randomized inputs w/ scaled flow
    #full randomizer
    for i in range(0,1):
        geom = []
        geom = mesh()
        
#        #***** input block randomizer *****
#        #rock properties
        geom.rock.size = 150.0 #m
        geom.rock.ResDepth = np.random.uniform(1250.0,1250.0) #6000.0 # m #!!!
        geom.rock.ResGradient = np.random.uniform(28.0,32.0) #50.0 #56.70 # C/km; average = 25 C/km
        geom.rock.ResRho = np.random.uniform(2700.0,2700.0) #2700.0 # kg/m3
        geom.rock.ResKt = np.random.uniform(2.1,2.8) #2.5 # W/m-K
        geom.rock.ResSv = np.random.uniform(1900.0,2200.0) #2063.0 # kJ/m3-K
        geom.rock.AmbTempC = np.random.uniform(20.0,20.0) #25.0 # C
        geom.rock.AmbPres = 0.101 #Example: 0.01 MPa #Atmospheric: 0.101 # MPa
        geom.rock.ResE = np.random.uniform(50.0,80.0)*GPa #50.0*GPa
        geom.rock.Resv = np.random.uniform(0.25,0.30) #0.3
        geom.rock.Ks3 = np.random.uniform(0.5,0.5) #0.5 #!!!
        geom.rock.Ks2 = geom.rock.Ks3 + np.random.uniform(0.4,0.6) # 0.75 #!!!
        geom.rock.s3Azn = 14.4*deg
        geom.rock.s3AznVar = 5.0*deg
        geom.rock.s3Dip = 27.0*deg
        geom.rock.s3DipVar = 5.0*deg
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
        geom.rock.fDia = np.asarray([[10.0,100.0],
                                [10.0,100.0],
                                [10.0,100.0]],dtype=float) #m
        #FORGE rotated 104 CCW
        geom.rock.fStr = np.asarray([[15.0*deg,7.0*deg],
                                [260.0*deg,7.0*deg],
                                [120.0*deg,7.0*deg,]],dtype=float) #m
        geom.rock.fDip = np.asarray([[35.0*deg,7.0*deg],
                                [69.0*deg,7.0*deg,],
                                [35.0*deg,7.0*deg]],dtype=float) #m
        #fracture hydraulic parameters
#        r = np.random.exponential(scale=0.25,size=2)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        r = r*(0.100/MPa-0.001/MPa)+0.001/MPa   
#        u1 = -np.min(r)
#        u2 = -np.max(r)
#        u3 = 0.5*(u1+u2)
#        geom.rock.alpha = np.asarray([u1,u3,u2])
        geom.rock.alpha = np.asarray([-0.028/MPa,-0.028/MPa,-0.028/MPa])
        
#        r = np.random.exponential(scale=0.25,size=2)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        r = r*(0.1-0.001)+0.001   
#        u1 = np.min(r)
#        u2 = np.max(r)
#        u3 = 0.5*(u1+u2)
#        geom.rock.gamma = np.asarray([u1,u3,u2])
        geom.rock.gamma = np.asarray([0.01,0.01,0.01])
        
        geom.rock.n1 = np.asarray([1.0,1.0,1.0])
        
#        r = np.random.exponential(scale=0.25,size=2)
#        r[r>1.0] = 1.0
#        r[r<0] = 0.0
#        r = r*(0.2-0.012)+0.012   
#        u1 = np.min(r)
#        u2 = np.max(r)
#        u3 = 0.5*(u1+u2)
#        geom.rock.a = np.asarray([u1,u3,u2])
        geom.rock.a = np.asarray([0.05,0.05,0.05])
        
#        u1 = np.random.uniform(0.7,0.9)
#        u2 = np.random.uniform(0.7,0.9)
#        u3 = 0.5*(u1+u2)
#        geom.rock.b = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
        geom.rock.b = np.asarray([0.8,0.8,0.8])
        
#        u1 = np.random.uniform(0.2,1.2)
#        u2 = np.random.uniform(0.2,1.2)
#        u3 = 0.5*(u1+u2)
#        geom.rock.N = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
        geom.rock.N = np.asarray([0.2,0.5,1.2])
        
#        u1 = np.random.uniform(0.00005,0.00015)
#        u2 = np.random.uniform(0.00005,0.00015)
#        u3 = 0.5*(u1+u2)
#        geom.rock.bh = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])])
        geom.rock.bh = np.asarray([0.00005,0.0001,0.003])
        
        geom.rock.bh_min = 0.00005 #m
        geom.rock.bh_max = 0.01 #0.02000 #m
#        geom.rock.bh_bound = np.random.uniform(0.001,0.005)
        geom.rock.bh_bound = 0.002
        geom.rock.f_roughness = np.random.uniform(0.25,1.0) #0.8
        #well parameters
#        geom.rock.w_count = int(np.random.uniform(1,4)) #2 #wells
#        geom.rock.w_spacing = np.random.uniform(100.0,800.0) #300.0 #m
#        geom.rock.w_length = np.random.uniform(800.0,1600.0) #800.0 #m
#        geom.rock.w_azimuth = geom.rock.s3Azn + np.random.uniform(-10.0,10.0)*deg #rad
#        geom.rock.w_dip = geom.rock.s3Dip + np.random.uniform(-10.0,10.0)*deg #rad
#        geom.rock.w_proportion = np.random.uniform(1.2,0.5) #0.8 #m/m
#        geom.rock.w_phase = np.random.uniform(0.0,360.0)*deg #-90.0*deg #rad
#        geom.rock.w_toe = np.random.uniform(0.0,0.0)*deg #rad
#        geom.rock.w_skew = np.random.uniform(-15.0,15.0)*deg #rad
#        geom.rock.w_intervals = int(np.random.uniform(2,3)) #3 #breaks in well length
#        geom.rock.ra = 0.0254*np.random.uniform(2.0,7.0) #3.0 #0.0254*3.0 #m
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
        geom.rock.ra = 0.0254*2.25 #0.0254*np.random.uniform(3.0,8.0) #3.0 #0.0254*3.0 #m
        geom.rock.rgh = 80.0
        #cement properties
        geom.rock.CemKt = 2.0 # W/m-K
        geom.rock.CemSv = 2000.0 # kJ/m3-K
        #thermal-electric power parameters
        geom.rock.GenEfficiency = 0.85 # kWe/kWt
        geom.rock.LifeSpan = 20.5*yr #years
        geom.rock.TimeSteps = 41 #steps
        geom.rock.p_whp = 1.0*MPa #Pa
        geom.rock.Tinj = 10.0 #95.0 #C
        geom.rock.H_ConvCoef = 3.0 #kW/m2-K
        geom.rock.dT0 = 10.0 #K
        geom.rock.dE0 = 500.0 #kJ/m2
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
        geom.rock.r_perf = 50.0 #m
        geom.rock.sand = 0.3 #sand ratio in frac fluid
        geom.rock.leakoff = 0.0 #Carter leakoff
#        geom.rock.dPp = -1.0*np.random.uniform(1.0,10.0)*MPa #-2.0*MPa #production well pressure drawdown
        geom.rock.dPp = -2.0*MPa #production well pressure drawdown
        geom.rock.dPi = 0.1*MPa #!!!
        geom.rock.stim_limit = 5
    #    geom.rock.Qinj = 0.01 #m3/s
        geom.rock.Qstim = 0.01 #0.08 #m3/s
        geom.rock.Vstim = 1000.0 #100000.0 #m3
        geom.rock.bval = 1.0 #Gutenberg-Richter magnitude scaling
#        u1 = np.random.uniform(20.0,55.0)*deg
#        u2 = np.random.uniform(20.0,55.0)*deg
#        u3 = 0.5*(u1+u2)
#        geom.rock.phi = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])]) #rad
        geom.rock.phi = np.asarray([25.0*deg,35.0*deg,45.0*deg]) #rad
#        u1 = np.random.uniform(5.0,20.0)*MPa
#        u2 = np.random.uniform(5.0,20.0)*MPa
#        u3 = 0.5*(u1+u2)
#        geom.rock.mcc = np.asarray([np.min([u1,u2]),u3,np.max([u1,u2])]) #Pa
        geom.rock.mcc = np.asarray([5.0,10.0,15.0])*MPa #Pa
        geom.rock.hfmcc = np.random.uniform(0.0,0.2)*MPa #0.1*MPa
        geom.rock.hfphi = np.random.uniform(15.0,45.0)*deg #30.0*deg
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
        
        #stimulate
        geom.dyn_stim(Vinj=geom.rock.Vstim,Qinj=geom.rock.Qstim,target=[],
                      visuals=False,fname='stim')
        
        #test multiple randomly selected flow rates
        #rates = np.random.uniform(0.005,0.08,10)
        rates = [0.007]
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
            x = base.save('inputs_results_valid.txt',pin)
    
    
    #show plots
    pylab.show()
        
        
# ****************************************************************************
#### test program
# ****************************************************************************          
if False: #test program
    geom = mesh()
#    geom.rock.s3Azn = 0.0*deg
#    geom.rock.s3Dip = 0.0*deg
#    phi = 30.0*deg
#    mcc = 0.0*MPa
#    stress = cauchy()
#    stress.set_sigG_from_Principal(geom.rock.s3, geom.rock.s2, geom.rock.s1, geom.rock.s3Azn, geom.rock.s3Dip)
#    stress.plot_Pc(phi,mcc)
#    print 'Pc for frac'
#    print stress.Pc_frac(90.0*deg,90.0*deg,phi,mcc)
    
    #generate domain
    geom.gen_domain()
    
#    #generate natural fractures
#    geom.gen_natfracs(f_num=8,
#                      f_dia = [1200.0,2400.0],
#                      f_azn = [79.0*deg,8.0*deg], #[0.0*deg,3000.0*deg],#[79.0*deg,8.0*deg],
#                      f_dip = [90.0*deg,12.5*deg]) #[90.0*deg,0.1*deg])#,12.5*deg])
#    #generate natural fractures
#    geom.gen_natfracs(f_num=80,
#                      f_dia = [300.0,900.0],
#                      f_azn = [79.0*deg,8.0*deg], #[0.0*deg,3000.0*deg],#[79.0*deg,8.0*deg],
#                      f_dip = [90.0*deg,12.5*deg]) #[90.0*deg,0.1*deg])#,12.5*deg])
    
    #generate natural fractures
    geom.gen_joint_sets()
    
    #generate wells
    wells = []
#    wells += [line(0.0-1.0*100.0,-300.0,0.0,600.0,0.0*deg,0.0*deg,'producer',0.2286,80.0)]
#    wells += [line(0.0+0.0*100.0,-300.0,0.0,600.0,0.0*deg,0.0*deg,'injector',0.2286,80.0)]
#    wells += [line(0.0+1.0*100.0,-300.0,0.0,600.0,0.0*deg,0.0*deg,'producer',0.2286,80.0)]
    geom.gen_wells(True,wells)
        
#    #generate hydraulic fracture
##    geom.dyn_stim(Vinj=100000.0,Qinj=0.08,dpp=-2.0*MPa,sand=0.3,leakoff=0.0,
##                  target=1,clear=True,visuals=False,fname='stim')
#    geom.dyn_stim(Vinj=100000.0,Qinj=0.08,
#                  target=1,clear=True,visuals=False,fname='stim')
#    
#    #calculate normal flow
#    geom.dyn_stim(Vinj=12614400.0,Qinj=0.02,
#                  target=1,clear=False,visuals=False,fname='stim')
    
    #calculate injection and production after stimulation
#    for i in range(0,len(geom.wells)):
#        if (int(geom.wells[i].typ) in [typ('injector')]):
#            target = int(i)
#            geom.stim_and_flow(target=i,visuals=False,fname='run')
    #random identifier (statistically should be unique)
    pin = np.random.randint(100000000,999999999,1)[0]
    
    geom.stim_and_flow(target=[],visuals=True,fname='run_%i' %(pin))
    
    #calculate heat transfer
    geom.get_heat(plot=True)
    
    #show flow model
    geom.build_vtk(fname='fin_%i' %(pin))
#    geom.rock.stress.plot_Pc(geom.rock.phi[1],geom.rock.mcc[1])
    
    #save primary inputs and outputs
    x = geom.save('inputs_results.txt',pin)
    
    #show plots
    pylab.show()
    
#    #calculate flow
#    bhp = geom.rock.BH_P #Pa
#    dpp=-2.0*MPa
#    pwp = bhp + dpp #Pa
#    geom.get_flow(p_bound=bhp,q_inlet=[0.02],p_outlet=[pwp])
#    
#    #build result vtk
#    geom.build_vtk('fin')

    
#    #calculate flow
#    geom.get_flow()
#        #standard fluid density
#        rho = self.rock.PoreRho #kg/m3
#        #bottom hole pressure, reservoir pore pressure
#        bhp = self.rock.BH_P #Pa
#        #production well pressure
#        pwp = bhp + dpp #Pa
#        #trial injection pressure
#        tip = self.rock.s3 + 6.0*MPa #Pa
#        #stim volume
#        vol = 0.0 #m3
#        
#        #looping variables
#        iters = 0
#        maxit = 1
#        while 1:
#            #***   loop breaker   ***
#            if iters >= maxit:
#                break
#            iters += 1
#            print '-> rock stimulation step %i' %(iters)
#            
#            #***   pressure based stimulation   ***
#            #solve flow with max pressure drive
#            self.get_flow(p_bound=bhp,p_inlet=[tip],p_outlet=[pwp])    
    
#    #initialize model
#    geom.re_init()
#    
#    #base parameter assignments
#    geom.static_KQn()
#    
#    #geom.gen_domain()
#    geom.build_vtk('test')



# ****************************************************************************
#### main program
# ****************************************************************************
if False: #main program
    #create mesh object (the model)
#    mnode = []
#    mpipe = []
#    mfrac = []
#    mwell = []
#    mhydf = []
#    mboun = []
#    mfrac = np.asarray(mfrac)
#    geo = [None,None,None,None,None] #[natfracs, origin, intlines, points, wells]
#    geo[0]=sg.mergeObj(geo[0], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0.1,0,0]), r=0.1))
#    geo[1]=sg.mergeObj(geo[1], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([20,0,0]), r=3.0))
#    geo[1]=sg.mergeObj(geo[1], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0,20,0]), r=3.0))
#    geo[1]=sg.mergeObj(geo[1], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0,0,20]), r=3.0))
#    geo[2]=sg.mergeObj(geo[2], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0.1,0,0]), r=0.1))
#    geo[3]=sg.mergeObj(geo[3], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0.1,0,0]), r=0.1))
#    geo[4]=sg.mergeObj(geo[3], sg.cylObj(x0=np.asarray([0,0,0]), x1=np.asarray([0.1,0,0]), r=0.1))
    geom = mesh() #node=mnode,pipe=mpipe,fracs=mfrac,wells=mwell,hydfs=mhydf,bound=mboun,geo3D=geo)
    
    #generate the domain
    geom.gen_domain()
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #modify reservoir parameters as needed
    geom.rock.size = 800.0 #m
    geom.rock.ResDepth = 6000.0 # m
    geom.rock.ResGradient = 50.0 #56.70 # C/km; average = 25 C/km
    geom.rock.LifeSpan = 20.0 # Years
    geom.rock.CasingIR = 0.0254*3.0 # m
    geom.rock.CasingOR = 0.0254*3.5 # m
    geom.rock.BoreOR = 0.0254*4.0 # m
    geom.rock.PoreRho = 960.0 #kg/m3 starting guess
    geom.rock.Poremu = 0.9*cP #Pa-s
    geom.rock.Porek = 0.1*mD #m2
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    #generate natural fractures
    geom.gen_natfracs(f_num=40,
                      f_dia = [300.0,1200.0],
                      f_azn = [79.0*deg,8.0*deg], #[0.0*deg,3000.0*deg],#[79.0*deg,8.0*deg],
                      f_dip = [90.0*deg,12.5*deg]) #[90.0*deg,0.1*deg])#,12.5*deg])
    
    #vary well spacing with N-S oriented wells
    spacing = [400.0] #[100.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0] #,300.0,400.0]
    w_r_Es = []
    w_i_Ps = []
    w_P_os = []
    w_p_hs = []
    w_p_ms = []
    first = True
    for s in range(0,len(spacing)):
        #generate wells
        wells = []
        #wells += [line(0.0+0.5*spacing[s],-300.0,0.0,600.0,0.0*deg,0.0*deg,'injector',0.2286,80.0)]
        #wells += [line(0.0-0.5*spacing[s],-300.0,0.0,600.0,0.0*deg,0.0*deg,'producer',0.2286,80.0)]
        wells += [line(0.0-1.0*spacing[s],-300.0,0.0,600.0,0.0*deg,0.0*deg,'producer',0.2286,80.0)]
        wells += [line(0.0+0.0*spacing[s],-300.0,0.0,600.0,0.0*deg,0.0*deg,'injector',0.2286,80.0)]
        wells += [line(0.0+1.0*spacing[s],-300.0,0.0,600.0,0.0*deg,0.0*deg,'producer',0.2286,80.0)]
#        #injection well
#        wells += [well(300.0,-200.0,0.0, 600.0, 324.0*deg, 0.0*deg,-1,0.2286,80.0)]
#        #production well
#        wells += [well(000.0,-400.0,0.0, 600.0, 324.0*deg, 0.0*deg,-2,0.2286,80.0)]
        geom.gen_wells(wells)
        
        #generate fractures
        geom.gen_stimfracs(target=1,perfs=2)

        #re-initialize the geometry - building list of faces
        geom.re_init()

        #populate fracture properties
        if first:
            first = False
            geom.static_KQn()

        #generate pipes
        geom.gen_pipes(plot=first)
        
        #test different flowrates
        flows = [-0.040] #[-0.001,-0.005,-0.010,-0.015,-0.020,-0.025,-0.030,-0.035,-0.040,-0.045,-0.050,-0.055,-0.060,-0.065]
        dpps = [-2.0*MPa]
        p_hs = []
        p_ms = []
        p_Es = []
        p_hms = []
        p_mms = []
        i_Ps = []
        i_ms = []
        b_hs = []
        b_ms = []
        b_Es = []
        r_Es = []
        P_os = []
        
        #calculate suitable production pressure
        h_bh = geom.rock.BH_P/(geom.rock.PoreRho*g) #m
        dpps = h_bh - np.asarray(dpps)/(geom.rock.PoreRho*g) #m
        
        for f in flows:
            #set boundary conditions (m3/s) (Pa); 10 kg/s ~ 0.01 m3/s for water
            geom.set_bcs(plot=False, 
                         p_bound=h_bh, 
                         q_inlet=[f], 
                         p_outlet=[dpps,dpps])
            
            #calculate flow
            geom.get_flow()
            
            #claculate heat transfer
            geom.get_heat(plot=False,t_n=21)
            p_hs += [geom.p_h]
            p_ms += [geom.p_m]
            p_hms += [geom.p_hm]
            p_mms += [geom.p_mm]
#            p_Es += [geom.p_E]
            i_Ps += [geom.i_p]
            i_ms += [geom.i_m]
            b_hs += [geom.b_h]
            b_ms += [geom.b_m]
#            b_Es += [geom.b_E]
            r_Es += [np.sum(geom.Et,axis=1)]
            P_os += [geom.Pout]
            
            
    #        #calculate power output
    #        geom.get_power(detail=False)
    #        P_os += [geom.Pout]
        
            #visualization
            fname = 'flow%.3f_spacing%.1f_well%i' %(-f,spacing[s],len(wells))
            geom.build_vtk(fname)
            
        #array format
        p_hs = np.asarray(p_hs)
        p_ms = np.asarray(p_ms)
        p_hms = np.asarray(p_hms)
        p_mms = np.asarray(p_mms)
#        p_Es = np.asarray(p_Es)
        i_Ps = np.asarray(i_Ps)
        i_ms = np.asarray(i_ms)
        b_hs = np.asarray(b_hs)
        b_ms = np.asarray(b_ms)
#        b_Es = np.asarray(b_Es)
        r_Es = np.asarray(r_Es)
        P_os = np.asarray(P_os)
        
        #store values for fancy plotting
        w_r_Es = [r_Es]
        w_i_Ps = [i_Ps]
        w_P_os = [P_os]
        w_p_hs = [p_hs]
        w_p_ms = [p_ms]    
                
        #plot key variables
        if True:
            fig = pylab.figure(figsize=(8.0, 6.0), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
            
            #mean production enthalpy
            ax1 = fig.add_subplot(221)
            for x in range(0,len(flows)):
                lab = '%.3f m3/s' %(flows[x])
                p_hm = p_hms[x]
                ax1.plot(geom.ts[:-1]/yr,p_hm[:-1],linewidth=0.5,label=lab)
            ax1.set_xlabel('Time (yr)')
            ax1.set_ylabel('Production Enthalpy (kJ/kg)')
            #ax1.legend(loc='upper right', prop={'size':8}, ncol=1, numpoints=1)
            
            #mean production energy
            ax2 = fig.add_subplot(222)
            for x in range(0,len(flows)):
                lab = '%.3f m3/s' %(flows[x])
                ax2.plot(geom.ts[:-1]/yr,P_os[x][:],linewidth=0.5,label=lab)
            ax2.set_xlabel('Time (yr)')
            ax2.set_ylabel('Production Energy (kJ/s)')
            #ax2.legend(loc='upper right', prop={'size':8}, ncol=1, numpoints=1)
            
            #max injection pressure
            ax3 = fig.add_subplot(223)
            i_pm = []
            for x in range(0,len(flows)):
                np.asarray(i_Ps)
                i_pm += [np.max(i_Ps[x,:])/MPa]
            ax3.plot(-np.asarray(flows),i_pm,'-',linewidth=0.5)
            ax3.set_xlabel('Injection Rate (m3/s)')
            ax3.set_ylabel('Injection Pressure (MPa)')
            #ax3.legend(loc='upper right', prop={'size':8}, ncol=1, numpoints=1)
            
            #total rock energy
            ax4 = fig.add_subplot(224)
            for x in range(0,len(flows)):
                lab = '%.3f m3/s' %(flows[x])
                ax4.plot(geom.ts/yr,r_Es[x],linewidth=0.5,label=lab)      
            ax4.set_xlabel('Time (yr)')
            ax4.set_ylabel('Rock Energy (kJ)')
            ax4.legend(loc='upper left', prop={'size':8}, ncol=2, numpoints=1)
            
            #per well performance
            NP = np.shape(p_hs)[1]
            for x in range(0,len(flows)):
                fig = pylab.figure(figsize=(8.0, 3.5), dpi=96, facecolor='w', edgecolor='k',tight_layout=True) # Medium resolution
                #production enthalpy
                ax1 = fig.add_subplot(121)
                for y in range(0,NP):
                    lab = 'P%i' %(y)
                    ax1.plot(geom.ts[:-1]/yr,p_hs[x,y,:-1],linewidth=0.5,label=lab)
                lab = 'Pnet'
                ax1.plot(geom.ts[:-1]/yr,p_hms[x,:-1],linewidth=1.0,label=lab)
                ax1.set_xlabel('Time (yr)')
                ax1.set_ylabel('Production Enthalpy (kg/kJ)')
                tit = 'Flow at %.3f m3/s' %(flows[x])
                ax1.set_title(tit)
                    
                #production energy
                ax2 = fig.add_subplot(122)
                for y in range(0,NP):
                    lab = 'P%i' %(y)
                    ax1.plot(geom.ts[:-1]/yr,p_hs[x,y,:-1],linewidth=0.5,label=lab)
                lab = 'Pnet'
                ax2.plot(geom.ts[:-1]/yr,p_hms[x,:-1],linewidth=1.0,label=lab)
                tit = 'Flow at %.3f m3/s' %(flows[x])
                ax2.set_title(tit)
                
                
            
    pylab.show()












