import numpy
from bilagrid import *
from scipy.ndimage.filters import gaussian_filter

def computeWeight(im, epsilonMini=0.002, epsilonMaxi=.99):
    out = numpy.ones(im.shape)
    #cond = out>epsilonMini and out<epsilonMaxi # try with 'and'
    out[(im<epsilonMini) | (im>epsilonMaxi)] = 0.0
    return out

def computeFactor(im1, w1, im2, w2):
    r = w1*w2*im1/im2
    r = r[r>0]
    r.flatten()
    r.sort()
    index =  int(r.shape[0]/2)
    return r[index]
    
def makeHDR(imageList, epsilonMini=.01, epsilonMaxi=.99):
    weight_sum = 0
    weight_comb = 0
    
    for i in range(len(imageList)):
        im_i = imageList[i]    
        w_i = computeWeight(im_i)
        if i == 0:  #darkest image, no preceding image
            k_i = 1.0
            w_i[im_i > epsilonMini] = 1.0
        else: # there is a preceding image
            im_im1 = imageList[i-1]
            w_im1 = computeWeight(im_im1)
            k_i *= computeFactor(im_i,w_i,im_im1,w_im1)
            if  i == len(imageList)-1: # brightest image
                w_i[(im_i>epsilonMini) & (im_i<epsilonMaxi)] = 1.0  # gotta use parens around conditions
            else:
                w_i[im_i<epsilonMaxi] = 1.0

        weight_comb += (w_i*im_i)/k_i 
        weight_sum += w_i
    
    weight_sum[weight_sum==0.0] = epsilonMini   # make sure we don't get NaN values
    return weight_comb/weight_sum
 
def toneMap(im, targetBase=100, detailAmp=3, useBila=False, maxLum = .99):
    lum, chrom = lumiChromi(im)
    h,w = im.shape[0:2]
    min = numpy.min(lum[lum>0])
    assert min>0 and targetBase>0
    
    lum[lum<=0] = min
    lum_log = numpy.log10(lum)
    sigmaS = getSigma(im)
    
    if useBila:
        sigmaR = .4
        bila = bilaGrid(lum_log[:,:,0], sigmaS, sigmaR)
        base = bila.doit(lum_log)
    else:
        base = numpy.zeros((h,w,3))
        gaussian_filter(lum_log, [sigmaS, sigmaS, 0], output=base)
        
    detail = lum_log - base  
    large_range = numpy.max(base)-numpy.min(base)
    
    scaled_base = (log10(targetBase)/large_range)*(base - numpy.max(base))
    
    log_lum = detail*detailAmp + scaled_base
    
    lum = numpy.power(log_lum, 10)
    return lum*chrom
    
      
    
           
def getSigma(im):
    h, w= im.shape[0:2]
    return max(h,w)/50.0

def BW(im, weights=[.3,.6,.1]):
    bw = numpy.dot(im, weights)
    return numpy.array([bw,bw,bw]).transpose(1,2,0)
    

def lumiChromi(im):
    lum = BW(im)
    lum[lum<=0] = .000001
    chrom = im/lum
    return (lum, chrom)   