import imageIO
import numpy as np
import math

def imIter(im):
    for y in xrange(im.shape[0]):
        for x in xrange(im.shape[1]): yield y, x

def blackAccessor(y, x, im):
    y_max, x_max = im.shape[0:2]
    if y>=y_max or x>=x_max or y<=0 or x<=0:
        return np.zeros([3])
    else:
        try:
            val = im[y,x]
            return val
        except IndexError as e:
            print y, x
            print e.args
            
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

def interpolateLin(y, x, im, accessor=blackAccessor):   #k_y, k_x better than y, x, which is confusing.
    y_avr= (accessor(y-1,x,im) + accessor(y+1,x,im))/2
    x_avr = (accessor(y,x-1,im) + accessor(y, x+1,im))/2
    pixel_vals = (y_avr + x_avr)/2
    return pixel_vals

def isInsideIm(y,x,im):
    h = im.shape[0]
    w = im.shape[1]
    y_range, x_range = range(h-1), range(w-1)
    return y in y_range and x in x_range
#
#def applyhomography(source, out, H, bilinear=False):    
#    for y,x in imIter(out):
#        coor_p = np.dot(H,np.array([y,x,1.0])) #(y_p, x_p, w_p)
#        y_p = int(coor_p[0]/coor_p[2])
#        x_p = int(coor_p[1]/coor_p[2])
#        
#        if isInsideIm(y_p, x_p, source):
#            if bilinear:
#                pixel_vals = interpolateLin(y_p, x_p, source, accessor=edgePaddingAccessor)
#            else:
#                pixel_vals = edgePaddingAccessor(y_p, x_p, source)
#            out[y, x] = pixel_vals
           
    
#def computehomography(listOfPairs):
#    assert len(listOfPairs)>3
#    listOfPairs = listOfPairs[0:4]
#    a_matrix = []
#    for pair in listOfPairs:
#        y = pair[0][0]
#        x = pair[0][1]
#        y_p = pair[1][0]
#        x_p = pair[1][1]
#        row1 = [y, x, 1.0, 0, 0, 0, -y*y_p, -x*y_p, -y_p]
#        a_matrix.append(row1)
#        row2 = [0.0, 0, 0, y, x, 1.0, -y*x_p, -x*x_p, -x_p]
#        a_matrix.append(row2)
#    
#    last_row = [0.0, 0, 0, 0, 0, 0, 0, 0, 1]
#    a_matrix.append(last_row)
#    b_vector = last_row
#    
#    a_m = np.array(a_matrix)
#    b_v = np.array(b_vector)
#    a_inv = np.linalg.inv(a_m)
#    coeffs = np.dot(a_inv, b_v)
#    
#    return np.reshape(coeffs, (3,3))
#    
#    
def computeTransformedBBox(im, H):
    # H im->im2
    h, w = im.shape[0], im.shape[1]
    corners = [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]
    
    y_min, x_min = None, None
    y_max, x_max = None, None    
    h_inv = np.linalg.inv(H)
    
    for y, x in corners:
        coor_p = np.dot(h_inv, np.array([y, x, 1.0]))   #(y_p, x_p, w_p)
        y_p = int(coor_p[0]/coor_p[2])
        x_p = int(coor_p[1]/coor_p[2])
        
        if y_p < y_min or y_min==None:
            y_min = y_p
        if y_p > y_max or y_max==None:
            y_max = y_p
        if x_p < x_min or x_min==None:
            x_min = x_p
        if x_p > x_max or x_max==None:
            x_max = x_p
            
    return np.array([[y_min, x_min],[y_max, x_max]])
#                     
#def bboxUnion(B1, B2):
#    union = np.concatenate(B1, B2)
#    y_min = np.amin(union, axis=0)
#    y_max = np.amax(union, axis=0)
#    x_min = np.amin(union, axis=1)
#    x_max = np.amax(union, axis=1)
#    return np.array([[y_min, x_min],[y_max, x_max]])
#
def translate(bbox):
    t_y, t_x = -bbox[0][0], -bbox[0][1]
    
    t_m = np.array([[1.0, 0, t_y],[0, 1, t_x],[0, 0, 1]])
    return t_m

def stitch(im1, im2, listOfPairs):
    H = computeHomography(listOfPairs)  # pairs from 1->2
    bbox = computeTransformedBBox(im1, H)
    t_m = translate(bbox)
    t_m_inv = np.linalg.inv(t_m)
    
    h, w = bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]
    out = imageIO.constantIm(h, w, [0, 0, 0.0])   
    applyHomography(im1, out, t_m, bilinear=True)
    comb = t_m+H
#    applyhomography(im2, out, comb, bilinear=True)
    return out
    
## FREDO's CODE ###########
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
    
    
    
    
    