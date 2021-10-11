""" Classes and routines for generating 3D objects
"""
import math
import numpy as np
from scipy.spatial import ConvexHull
from copy import deepcopy

# If you have placed the other modules in another directory, this can be useful to add that location to the path
#import os, sys; sys.path.append(os.path.dirname(__file__)); import Units as unit

# print( 'Loaded SimpleGeometry from same level')

def dipDirectionAndDipAng(tangent):
    # Given the tangent to a curve, convert into dip direction and dip
    e=tangent[0]
    n=tangent[1]
    up=tangent[2]
    # If we rotate compass to align with math coords:
    #        W
    #        |
    #    S-------N
    #        |
    #        E
    x=n
    y=-e
    thetaMath=math.atan2(y,x)
    thetaCompass=-thetaMath*180.0/math.pi
    dipDirection=thetaCompass
    # Dip angle is the amount we are dipping from horizontal
    # We chose orientation such that up is -ve
    dipAngle=math.atan2( -up, math.sqrt( e*e + n*n ) )*180.0/math.pi
    return dipDirection,dipAngle
    
def dipToStrikeDeg(dip_dir_deg):
    # According to Wikipedia:
    # https://en.wikipedia.org/wiki/Strike_and_dip
    # One technique is to always take the strike so the dip is 90 deg to the right of the strike, in which case the redundant letter following the dip angle is omitted (right hand rule, or RHR).
    #strike_rad=dip_dir_radians-0.5*np.pi
    strike_deg=dip_dir_deg-90.0
    return strike_deg

def degToRad(deg):
    return deg*np.pi/180.0

def radToDeg(rad):
    return rad*180.0/np.pi
    
def writeStlObject(points,simplices,fd):
    for simplex in simplices:
        # I'm not sure we need to calculate a normal...
        fd.write("facet normal 0.0 0.0 0.0\n")
        fd.write("outer loop\n")
        for iPt in simplex:
            #print iPt,simplex
            fd.write("vertex %g %g %g\n"%(points[iPt][0],points[iPt][1],points[iPt][2]))
        fd.write("endloop\n")
        fd.write("endfacet\n")
    
def writeStlFile(points,simplices,stlFile,name="stlObject"):
    fd=open(stlFile,'w')
    fd.write("solid %s\n"%(name))
    writeStlObject(points,simplices,fd)
    fd.write("endsolid\n");

def writeObjectsStlFile(objects,stlFile,name="stlObject"):
    fd=open(stlFile,'w')
    fd.write("solid %s\n"%(name))
    #for object in objects:
    (points,simplices)=objects
    writeStlObject(points,simplices,fd)
    fd.write("endsolid\n");

def writeVtk(objectList,scalars,scalarNames,vtkFile,name="vtkObjects"):

    fd=open(vtkFile,'w')

    nPtsObj=[]
    nPts=0
    nTri=0
    nObj=len(objectList)
    for pts,simps in (objectList):
        nPtsObj.append(len(pts))
        nPts+=len(pts)
        nTri+=len(simps)
    nShift=[0]*nObj
    for iShift in range(nObj-1):
        nShift[iShift+1]=nShift[iShift]+nPtsObj[iShift]
    
    fd.write("# vtk DataFile Version 2.0\n")
    fd.write("%s\n"%(name))
    fd.write("ASCII\n")
    fd.write("DATASET UNSTRUCTURED_GRID\n")
    fd.write("POINTS %d float\n"%(nPts))
    for pts,simps in (objectList):
        for pt in (pts):
            fd.write("%g %g %g\n"%(pt[0],pt[1],pt[2]))
    
    fd.write("CELLS %d %d\n"%(nTri,(1+3)*nTri))
    iObj=0
    #col=[]
    for pts,simps in (objectList):
        for tri in (simps):
            fd.write("3 %d %d %d\n"%(tri[0]+nShift[iObj],tri[1]+nShift[iObj],tri[2]+nShift[iObj]))
            #col.append(colorList[iObj])
        iObj+=1
            
    fd.write("CELL_TYPES %d\n"%(nTri))
    for i in range(nTri):
        fd.write("5 ") # http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf (see Fig. 2)
        if (i%10==9):
            fd.write("\n")
    fd.write("\n")

    fd.write("CELL_DATA %d\n"%(nTri))


    # Repeat as many of these as you want to define data on the tris
    #for colorList, scalarName in scalars,scalarNames:
    for iCol in range(len(scalars)):
        colorList=scalars[iCol]; scalarName=scalarNames[iCol]
        fd.write("SCALARS "+scalarName+" float 1\n")
        fd.write("LOOKUP_TABLE default\n")
        iObj=0
        i=0
        for pts,simps in (objectList):
            for tri in (simps):
                fd.write("%g "%(colorList[iObj])); i+=1
                if (i%10==9):
                    fd.write("\n")
            iObj+=1
    fd.write("\n")
    fd.close()
    
    
def simplicesFromPoints(points):
    hull=ConvexHull(points)
    return hull.simplices

def convexFromPoints(points):
    return ( points, simplicesFromPoints(points) )
    
# A non-object
emptyObject=None

# Merging two objects requires a shift in the indices
def mergeObj(obj1, obj2):
    if (obj1==emptyObject):
        return obj2
    if (obj2==emptyObject):
        return obj1
    return (
        np.vstack( (obj1[0],obj2[0]) ),
        np.vstack( (obj1[1],obj2[1]+len(obj1[0])) )
    )

def mergeObjects(objects):
    nObj=len(objects)
    merged=np.asarray(deepcopy(objects[0]))
    nShift=0
    for i in range(nObj-1):
        #print i
        nShift+=len(objects[i][0])
        merged[0]=np.vstack( (merged[0],objects[i+1][0]) )
        merged[1]=np.vstack( (merged[1],objects[i+1][1]+nShift) )
    return merged
    
# Some useful objects
unitCubePts=np.asarray([
    [-0.5,-0.5,-0.5],
    [ 0.5,-0.5,-0.5],
    [-0.5, 0.5,-0.5],
    [ 0.5, 0.5,-0.5],
    [-0.5,-0.5, 0.5],
    [ 0.5,-0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [ 0.5, 0.5, 0.5]
    ])
Cube=convexFromPoints(unitCubePts)
unitWedgePts=np.asarray([
    [-0.5,-0.5,-0.5],
    [ 0.5,-0.5,-0.5],
    [ 0.0, 0.5,-0.5],
    [-0.5,-0.5, 0.5],
    [ 0.5,-0.5, 0.5],
    [ 0.0, 0.5, 0.5]
    ])
unitWedge=convexFromPoints(unitWedgePts)

def diskObj(r, h, n=50):
    dTh=2*math.pi/n
    pts=[]
    for i in range(n):
        x=r*math.cos(i*dTh); y=r*math.sin(i*dTh)
        pts.append( [x,y,-0.5*h] )
        pts.append( [x,y, 0.5*h] )
    pts=np.asarray(pts)
    return convexFromPoints(pts)

# From: https://en.wikipedia.org/wiki/Regular_dodecahedron
# Golden ratio
gr=(1.0+math.sqrt(5.0))/2.0
radiusOneSpherePts=np.asarray([
    [-1,-1,-1],[ 1,-1,-1], [-1, 1,-1],[ 1, 1,-1], [-1,-1, 1],[ 1,-1, 1], [-1, 1, 1],[ 1, 1, 1],
    [0,-1/gr,-gr],[0, 1/gr,-gr],[0,-1/gr, gr],[0, 1/gr, gr],
    [-1/gr,-gr,0],[ 1/gr,-gr,0],[-1/gr, gr,0],[ 1/gr, gr,0],
    [-gr,0,-1/gr],[-gr,0, 1/gr],[ gr,0,-1/gr],[ gr,0, 1/gr]
    ])
radiusOneSphereObj=convexFromPoints(radiusOneSpherePts)

def cylObj(x0, x1, r, n=10, lengthSum=None):
    sphere0=(r*radiusOneSpherePts)
    sphere0[:,0]+=x0[0]; sphere0[:,1]+=x0[1]; sphere0[:,2]+=x0[2];
    sphere1=(r*radiusOneSpherePts)
    sphere1[:,0]+=x1[0]; sphere1[:,1]+=x1[1]; sphere1[:,2]+=x1[2];
    pts=np.vstack( (sphere0, sphere1) )
    #print lengthSum
    #if (lengthSum != None):
    try:
        lengthSum[0]+=np.sqrt( np.dot((x1-x0),(x1-x0)) )
    except:
        pass
    #print lengthSum
    return convexFromPoints(pts)

# Set up a unit arrow pointing in y-direction
pts1=deepcopy(unitCubePts); pts1[:,1]-=0.5
pts2=deepcopy(unitWedgePts); pts2[:,0]*=2.0; pts2[:,1]+=0.5
unitArrow1=convexFromPoints(pts1)
unitArrow2=convexFromPoints(pts2)
unitArrowY=mergeObj(unitArrow1,unitArrow2)

def extrudePoints(points, disp):
    """
    Return a list of points including the initial points and extruded end
    """
    farEnd=deepcopy(points)
    farEnd[:,0]+=disp[0]
    farEnd[:,1]+=disp[1]
    farEnd[:,2]+=disp[2]
    return np.vstack( (points,farEnd) )

def transObj(object, disp):
    """
    Translate an object
    """
    return (object[0]+disp,object[1])

def scaleObj(object, scale):
    """
    Scale an object
    """
    return (object[0]*scale,object[1])


# http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
def rotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotatePoints(points, axis, theta):
    rot=rotationMatrix(axis,theta)
    #return rot*points
    return np.transpose( np.dot(rot, np.transpose( points ) ) )

def rotateTensor(tensor, axis, theta):
    #http://www.continuummechanics.org/stressxforms.html
    rot=rotationMatrix(axis,theta)
    return np.dot(rot,np.dot(tensor,np.transpose(rot)))
    
def rotateObj(object, axis, theta):
    rot=rotationMatrix(axis,theta)
    return ( np.transpose( np.dot(rot, np.transpose(object[0])) ) , object[1])

# Taken from:
# http://geomalgorithms.com/a05-_intersect-1.html
def intersectionOfLineAndPlane(lineX,lineS,planeX,planeN):
    V0=np.asarray(planeX)
    n=np.asarray(planeN)
    P0=np.asarray(lineX)
    u=np.asarray(lineS)
    sI=( np.dot( n, (V0-P0) ) )/( np.dot( n,u ) )
    return P0+sI*u

# From math.stackexchange.com find-shortest-distance-between-lines-in-3d
def shortestDistanceBetweenLines(a,b, c,d):
    # a=origin of first line
    # b=tangent to first line
    # c=origin of second line
    # d=tangent to second line
    # print "a",a
    # print "b",b
    # print "c",c
    # print "d",d

    # t=path length along first line
    # s=path length along second line

    e=a-c

    A = -np.dot(b,b)*np.dot(d,d) + np.dot(b,d)*np.dot(b,d)

    s = ( -np.dot(b,b)*np.dot(d,e) + np.dot(b,e)*np.dot(d,b) )/A
    t = (  np.dot(d,d)*np.dot(b,e) - np.dot(b,e)*np.dot(d,b) )/A

    # print "s",s
    # print "t",t
    
    dvect=e+b*t-d*s

    # print "dvect",dvect
    
    dist=np.sqrt( np.dot( dvect, dvect ) )

    return dist

# Place a radial hydraulic fracture of radius r at x0
def HF(r,x0, strikeRad, dipRad, h=0.5):
    # start with a disk
    disk=diskObj(r,h)
    disk=rotateObj(disk,[0.0,1.0,0.0],dipRad)
    disk=rotateObj(disk,[0.0,0.0,1.0],-strikeRad)
    disk=transObj(disk,x0)
    return disk
