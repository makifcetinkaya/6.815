import imageIO
import numpy as np
import numpy
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.signal import convolve
import math
import random as rnd

SobelX = np.array([[-1, 0, 1],[-2, 0, 2],[-1,0,1]])
SobelY = np.transpose(SobelX)

def imIter(im):
    for y in xrange(im.shape[0]):
        for x in xrange(im.shape[1]): yield y, x

def BW(im, weights=np.array([.3,.6,.1])):
    bw = np.dot(im,weights)
    return bw

def edgePaddingAccessor(y, x, im): 
    assert isinstance(y, int) and isinstance(x, int)
    h,w = im.shape[0:2]
    y_range, x_range = range(h-1), range(w-1)
    if y not in y_range:
        if y > h:
            y = h-1
        else:
            y = 0
    if x not in x_range:
        if x >= w:
            x = w-1
        else:
            x = 0
    return im[y,x]


def computeTensor(im, sigmaG=1, factorSigma=4):
    lum = np.dot(im, np.array([.3,.6,.1]))
    lum = np.sqrt(lum)
    
    blurred = gaussian_filter(lum, sigmaG, mode='nearest')
    
    i_x = convolve(blurred, SobelX, mode='same')
    i_y = convolve(blurred, SobelY, mode='same')
      
    M_11 = gaussian_filter(i_x*i_x, sigma=sigmaG*factorSigma, mode='nearest')
    M_22 = gaussian_filter(i_y*i_y, sigma=sigmaG*factorSigma, mode='nearest')
    M_12 = gaussian_filter(i_x*i_y, sigma=sigmaG*factorSigma,  mode='nearest')
    M = np.array([M_11,M_12,M_22])
    
    return M
    
    
def HarrisCorners(im, k=.15, sigmaG=1, factor=4, maxiDiam=7, boundarySize=5):
    tensor = computeTensor(im, sigmaG=sigmaG, factorSigma=factor)
    # transpose so that M components will be in 0th dimension - for convenience
    #tensor = np.transpose(t, (2,0,1))
    
    m = np.array([[tensor[0], tensor[1]],[tensor[1],tensor[2]]])
    M = np.transpose(m, (2, 3, 0, 1))
    
    # calculate corner response of each pixel
    corner_resp = np.zeros(im.shape[0:2])    
    for y, x in imIter(im):
        M_pix = M[y][x]
        eig_vals = np.linalg.eig(M_pix)[0]
        corner_resp[y][x] = eig_vals[0]*eig_vals[1] - k*((eig_vals[0]+eig_vals[1])**2)
    
    # now let's filter the image
    maxima = maximum_filter(corner_resp, size=maxiDiam, mode='nearest')
    # if maxima == corner_resp, keep corner_resp, or set to 0 because it is not a local maximum
    corner_resp = np.where(maxima==corner_resp, corner_resp, 0)
    zeroPadding(corner_resp, boundarySize)
    
    indices = np.transpose(np.nonzero(corner_resp))
    return indices
        
def computeFeatures(im, cornerL=None, sigmaBlurDescriptor=.5, radiusDescriptor=4):   
    cornerL = HarrisCorners(im)
    blurredIm = gaussian_filter(im, sigmaBlurDescriptor, mode='nearest')
    lum = BW(blurredIm)
    
    features = []
    for corner in cornerL:
        y, x = corner[0:2]
        # assume that no corner is closer to edges than radiusDescriptor pixels
        patch = lum[y-radiusDescriptor:y+1+radiusDescriptor, x-radiusDescriptor:x+1+radiusDescriptor]
        patch_mean = np.mean(patch)
        patch = patch - patch_mean
        patch_std = np.std(patch)
        patch = patch/patch_std
        features.append((corner, patch))
    return features
        
def findCorrespondences(LF1, LF2, threshold=1.7):
    cornerL1, patchL1  = zip(*LF1)    
    cornerL2, patchL2 = zip(*LF2) 
    
    corrL = []
    for i in range(len(patchL1)):
        patch1 = patchL1[i]
        corner1 = cornerL1[i]
        corrP2 = getPlausibleCorrespondence(patch1, patchL2, cornerL2, threshold)
        if corrP2 is not None:
            corrL.append((corner1,corrP2))#            
    return corrL

# returns the point of match that meets threshold criteria, or None if it does not.
def getPlausibleCorrespondence(patch1, patchL2, cornerL2, threshold):
    best = float('inf')
    sec_best = float('inf')
    best_corr = None
    
    for i in range(len(patchL2)):
        diff = patchL2[i] - patch1 
        diff_1d = np.reshape(diff, 81)
        dot = np.dot(diff_1d, diff_1d)
        new_dist = np.sqrt(dot)
        if new_dist < best:
            if sec_best == float('inf'):    # if sec_best not assigned yet, assign the new_dist
                sec_best = new_dist
            else:
                sec_best = best
                
            best = new_dist
            best_corr = cornerL2[i]
            
    
    ratio = sec_best/best
    if ratio > threshold:# 
        return best_corr
    else:
        return None

def RANSAC(listOfCorrespondences, Niter=1000, epsilon=4):   
    list_len = len(listOfCorrespondences)
    pL1, pL2 = zip(*listOfCorrespondences)  
    # initialize variables
    num_of_inliers = 0
    
    def isInlier(diff):
        if diff <epsilon:
            return True
        else:
            return False
        
    for i in range(Niter):
        maybe_inliers = rnd.sample(listOfCorrespondences, 4)
        # TO CHECK
        h = computeHomography(maybe_inliers) # from 1->2 
        
        homoedL1 = map(applyHP2d, [h]*list_len, pL1)
        diffL = map(lambda x,y:math.sqrt(np.dot(x-y,x-y)), pL2, homoedL1 )
        inliers = map(isInlier, diffL)
        
        n = inliers.count(True)
        
        if n > num_of_inliers:   # must get in here at least once, so no need to initialize below vars
            print "updating num of inliers"
            num_of_inliers = n
            chosen_h = h
            isinlierL = inliers
            
    return (chosen_h, isinlierL)
        
def autostitch(im1, im2, blurDescriptor=.5, radiusDescriptor=4):
    LF1 = computeFeatures(im1)       
    LF2 = computeFeatures(im2)
    corrL = findCorrespondences(LF1, LF2)
    H, isInlierL = RANSAC(corrL, Niter=200, epsilon=2)
    out = merge2images(im1, im2, H)
    return out
    

def zeroPadding(corner_resp, boundarySize):
    h,w = corner_resp.shape[0:2]
    corner_resp[0:boundarySize,:] = 0.0
    corner_resp[h-boundarySize:,:] = 0.0
    corner_resp[:,0:boundarySize] = 0.0
    corner_resp[:,w-boundarySize:] = 0.0
    
################### FREDO's CODE on PIAZZA #################

def linearReconstructP(im, P):
    rx, ix=math.modf(P[1])
    ry, iy=math.modf(P[0])
    vup=edgePaddingAccessor(iy, ix, im)*(1.0-rx)+edgePaddingAccessor(iy, ix+1, im)*rx
    vdown=edgePaddingAccessor(iy+1, ix,im)*(1.0-rx)+edgePaddingAccessor(iy+1, ix+1, im)*rx
    return vup*(1.0-ry)+vdown*ry

def homogenize(P3D):
    if (P3D[2]!=0):
        return np.array([P3D[0]/P3D[2], P3D[1]/P3D[2]])
    else: return np.array([0, 0])
        
def applyH3d(H, P):
    return homogenize(np.dot(H, P))

def applyH2d(H, y, x):
    return homogenize(np.dot(H, np.array([y, x, 1.0])))

def applyHP2d(H, P):
    return homogenize(np.dot(H, np.array([P[0], P[1], 1.0])))

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
    A=np.zeros([N, 9])
    B=np.zeros([N])
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
    X=np.dot(np.linalg.pinv(A), B.T)
    H=np.reshape(X, [3, 3])        
    return H

 
def computeTransformedBBox(im, H):
    y,x=im.shape[0]-1, im.shape[1]-1
    P=np.array([applyH2d(H, 0,0), applyH2d(H, y,0), applyH2d(H, 0,x), applyH2d(H, y,x)] )
    return np.array([numpy.min(P,0), numpy.max(P,0)])

def bboxUnion(B1, B2):
    return np.array([np.min([B1[0], B2[0]], 0), np.max([B1[1], B2[1]], 0)])

def translate(y, x):
    out=np.identity(3)
    out[0, 2]=y
    out[1, 2]=x
    return out


def merge2images(im1, im2, H):
    B=computeTransformedBBox(im1, np.linalg.inv(H))
    B=bboxUnion(B, np.array([[0.0, 0.0], 1.0*np.array(im2.shape[0:2])]))
    print 'bbox after union: ', B

    T=translate(B[0, 0], B[0, 1])
    
    out = imageIO.constantIm(B[1, 0]-B[0, 0], B[1, 1]-B[0, 1], 0)
    
     
    applyHomography(im1, out, T)
    applyHomography(im2, out, np.dot(H, T)) 
    return out
     
    