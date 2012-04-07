import imageIO
from imageIO import *
import numpy
from numpy import *
import scipy
from scipy import ndimage


def linearReconstructP(im, P):
    rx, ix=math.modf(P[1])
    ry, iy=math.modf(P[0])
    vup=getSafePix(im, iy, ix)*(1.0-rx)+getSafePix(im, iy, ix+1)*rx
    vdown=getSafePix(im, iy+1, ix)*(1.0-rx)+getSafePix(im, iy+1, ix+1)*rx
    return vup*(1.0-ry)+vdown*ry

def homogenize(P3D):
    if (P3D[2]!=0):
        return array([P3D[0]/P3D[2], P3D[1]/P3D[2]])
    else: return array([0, 0])
        
def applyH3d(H, P):
    return homogenize(dot(H, P))

def applyH2d(H, y, x):
    return homogenize(dot(H, array([y, x, 1.0])))

def applyHP2d(H, P):
    return homogenize(dot(H, array([P[0], P[1], 1.0])))

def isInside(P, im):
    return (0<P[0]<im.shape[0]) and (0<P[1]<im.shape[1])

def applyHomography(source, out, H, bilinear=False):
    if bilinear:
        for y, x, in imIter(out):
            P=applyH2d(H, y, x) 
            if isInside(P, source):
                out[y, x]=linearReconstructP(source, P)
    else:
        for y, x, in imIter(out):
            P=applyH2d(H, y, x) 
            if isInside(P, source):
                out[y, x]=source[P[0], P[1]]

def computeHomography(listOfPairs):
    N=2*len(listOfPairs)+1
    A=zeros([N, 9])
    B=zeros([N])
    A[N-1,8]=1
    B[N-1]=1
    def addpair(P1, P2, i):
        A[2*i,0]=P1[0]
        A[2*i,1]=P1[1]
        A[2*i, 2]=1
        A[2*i, 6]=-P2[0]*P1[0]
        A[2*i, 7]=-P2[0]*P1[1]
        B[2*i]=P2[0]
        
        A[2*i+1,3]=P1[0]
        A[2*i+1,4]=P1[1]
        A[2*i+1, 5]=1
        A[2*i+1, 6]=-P2[1]*P1[0]
        A[2*i+1, 7]=-P2[1]*P1[1]
        B[2*i+1]=P2[1]
    for i in xrange(len(listOfPairs)):
        addpair(listOfPairs[i][0], listOfPairs[i][1], i)
    X=dot(linalg.pinv(A), B.T)
    H=reshape(X, [3, 3])        
    return H
        
