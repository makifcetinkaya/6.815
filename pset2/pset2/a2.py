import numpy as np
from numpy.lib.scimath import sqrt

import imageIO 
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

def scaleNN(im, k):
    h, w = im.shape[0:2]
    out = imageIO.constantIm(h*k, w*k, 0.0)
    for y,x in imIter(out):
        try:
            y_orig, x_orig = int(y/k), int(x/k)
            out[y,x] = im[y_orig, x_orig]
        except: #only for testing purposes
            print y_orig, x_orig
            
    return out

def scaleLin(im, k):
    out = scaleNN(im, k)
    return interpolateLin(out, 1,1, accessor=edgePaddingAccessor)

def interpolateLin(im, k_y, k_x, accessor=blackAccessor):   #k_y, k_x better than y, x, which is confusing.
    h, w = im.shape[0:2]
    out = imageIO.constantIm(h, w, 0.0)
    for y,x in imIter(im):
        y_avr= (accessor(y-1,x,im) + accessor(y+1,x,im))/2
        x_avr = (accessor(y,x-1,im) + accessor(y, x+1,im))/2
        pixel_vals = (y_avr*k_y + x_avr*k_x)
        out[y,x] = pixel_vals
    return out

def getUV(segment, pointX):
    px = pointX-segment.P
    u = np.vdot(px, segment.delta)/ (segment.magnitude)**2
    v = np.vdot(px, segment.rotate90())/segment.magnitude
    return u,v

def transform(dest_point, dest_segment, src_segment):
    u,v = getUV(dest_segment, dest_point)
    src_point =  src_segment.P + u*(src_segment.delta) + \
        (v/src_segment.magnitude)*(src_segment.rotate90())
    #print "transformed point: " + str(t)
    return np.array(src_point, dtype=np.int64)
 
def warpBy1(im, src_segment, dest_segment):
    h,w = im.shape[0:2]
    out = imageIO.constantIm(h, w, [0, 0, 0])
    for y, x in imIter(out):
        dest_point = np.array([y,x])
        y_p, x_p = transform(dest_point, dest_segment, src_segment)
        out[y,x] = edgePaddingAccessor(y_p, x_p, im)
    return interpolateLin(out, .5,  .5, accessor=edgePaddingAccessor)
 
def warp(im, listSegmentsBefore, listSegmentsAfter, a=10.0, b=1.0, p=1.0):
    h,w = im.shape[0:2]
    out = imageIO.constantIm(h, w, [0,0,0])     
    for y,x in imIter(out):
        dest_point = np.array([y,x])
        dist_sum = np.array([0,0], dtype=np.float64)
        weight_sum = 0
        for i in range(len(listSegmentsBefore)):
            src_segment = listSegmentsBefore[i]
            dest_segment = listSegmentsAfter[i]
            
            src_point = transform(dest_point, dest_segment, src_segment)
            disp = src_point - dest_point
            dist = getShortestDist(dest_point, dest_segment)
            weight = (((dest_segment.magnitude)**p)/(a+dist))**b
            dist_sum += disp*weight
            weight_sum += weight
        y_p, x_p = dest_point + dist_sum/weight_sum
        out[y, x] = edgePaddingAccessor(int(y_p), int(x_p), im)
        
    return interpolateLin(out, .5, .5, accessor=edgePaddingAccessor)  
        
            
def getShortestDist(pointX, segment):
    u,v = getUV(segment, pointX) 
    if u<1 and u>0:
        return abs(v)
    elif u<0:
        px = segment.P - pointX
        return sqrt(np.vdot(px, px))
    else:
        qx = segment.Q - pointX
        return sqrt(np.vdot(qx, qx))
        
def morph(im1, im2, listSegmentsBefore, listSegmentsAfter, N=1, a=10, b=1, p=1): 
    assert im1.shape == im2.shape
    imageIO.imwrite(im1, 'im_start.png')
    imageIO.imwrite(im2, 'im_end.png') 
    delta_segments = listSegmentsAfter - listSegmentsBefore   
    for i in range(N):
        t = float(i+1)/(N+1)
        segments_t = listSegmentsBefore + delta_segments*t
        
        im1_t = warp(im1, listSegmentsBefore, segments_t, a, b, p)
        im2_t = warp(im2, listSegmentsAfter, segments_t, a, b, p)
        im_t = im1*(1-t)+im2*t
        imageIO.imwrite(im_t, 'morph'+str('%03d'%i)+'.png')
             
class segment():
    def __init__(self, x1, y1, x2, y2):
        self.P = np.array([y1, x1], dtype=np.float64)
        self.Q = np.array([y2, x2], dtype=np.float64)
        self.delta = self.Q - self.P        
        self.magnitude = sqrt(np.vdot(self.delta, self.delta))
      
    def rotate90(self):# change this to return to a segment
        return np.array([-self.delta[1], self.delta[0]])
        
    def __add__(self, other):
        new_p = self.P+other.P
        new_q = self.Q+other.Q
        y1, x1 = new_p[0:2]
        y2, x2 = new_q[0:2]
        return segment(x1, y1, x2, y2)  
    
    def __sub__(self, other):
        new_p = self.P-other.P
        new_q = self.Q-other.Q
        y1, x1 = new_p[0:2]
        y2, x2 = new_q[0:2]
        return segment(x1, y1, x2, y2) 
    def __mul__(self,k):
        new_p = self.P*k
        new_q = self.Q*k
        y1, x1 = new_p[0:2]
        y2, x2 = new_q[0:2]
        return segment(x1, y1, x2, y2) 
        
        