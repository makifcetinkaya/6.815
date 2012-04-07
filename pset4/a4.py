from imageIO import *
import numpy

def iter2D(y,x):
    for j in range(y):
        for i in range(x):
            yield y,x
def denoiseSeq(imageList):
    im = imageList[0]
    h,w = im.shape[0:2]
    mean = constantIm(h,w,[0.0,0.0,0.0])
    
    for im in imageList:
        mean += im
        
    return mean/len(imageList)

def getVariance(imageList, mean=None):
    variance = numpy.zeros(imageList[0].shape)
    if mean==None:
        mean = denoiseSeq(imageList)
    N = len(imageList)
    
    for im in imageList:
        diff = im - mean
        variance += diff*diff # how to multipy
    if N >1:
        variance /= (N-1)
    else: #the list has only one element
        variance /= N
    return variance

def meanSq(imageList):
    im = imageList[0]
    h,w = im.shape[0:2]
    meansq = constantIm(h,w,[0.0,0.0,0.0])
    
    for im in imageList:
        meansq += im*im
        
    return meansq/len(imageList)

def logSNR(imageList, scale=1.0/20.0):
    variance = getVariance(imageList)
    meansq = meanSq(imageList)
    var_y, var_x, var_p = numpy.where(variance==0)
    assert numpy.size(var_y) == numpy.size(var_x) # for testing
    mean_y, mean_x, mean_p = numpy.where(meansq==0) 
    #print "zeros: ", var_y, mean_y
    # get rid of zeros for safe division and logarithym
    for i in range(numpy.size(var_y)):
        variance[var_y[i]][var_x[i]][var_p[i]] =  0.000001
    for i in range(numpy.size(mean_y)):
        meansq[mean_y[i]][mean_x[i]][mean_p[i]] = 0.000001
    
    var_y, var_x, var_p = numpy.where(variance==0)
    assert len(var_y) == 0
    mean_y, mean_x, mean_p = numpy.where(meansq==0) 
    assert len(mean_y) == 0
        
    return scale*numpy.log10(meansq/variance)
    

def align(im1, im2, maxOffset=20): # try to match im2 to im1
    assert im1.shape == im2.shape
    h,w = im2.shape[0:2]
    assert maxOffset<h and maxOffset<w
    
    variances = []
    min_var_sum = None
    for j in range(maxOffset*2+1):
        for i in range(maxOffset*2+1):
            j_s = j - maxOffset
            i_s = i - maxOffset
            
            im2test = numpy.roll(im2, j_s, axis=0)
            im2test = numpy.roll(im2test, i_s, axis=1)
            
            delta = im1 - im2test
            variance = delta*delta
            #print variance
            
            # set var pixes less than maxOFfset far from edges to 0
            variance[:maxOffset,:, :] = 1.0
            variance[h-maxOffset-1:,:, :] = 1.0
            variance[:, :maxOffset, :] = 1.0
            variance[:, :w-maxOffset-1, :] = 1.0
            
            if(min_var_sum == None):
                min_var_sum = numpy.sum(variance)
                offsets = (j_s,i_s)
            else:
                if numpy.sum(variance) < min_var_sum:
                    min_var_sum = numpy.sum(variance)
                    offsets = (j_s,i_s)
    
    print "offsets: ", offsets
    return offsets      
            
def alignAndDenoise(imageList, maxOffset=20):  
    im1 = numpy.copy(imageList[0])
    # first align the images
    newList = [im1]
    for im in imageList:
        offsets = align(im1, im, maxOffset)
        new_im = numpy.roll(im, offsets[0], 0)
        new_im = numpy.roll(new_im, offsets[1],1)
        
        newList.append(new_im)
    
    return denoiseSeq(newList)

def basicGreen(raw, offset=1):
    h,w = raw.shape[0:2]
    out = numpy.zeros([h,w])
    for j in range(1, h-1):
        for i in range(1, w-1):
            if (j%2 == 0 and i%2 != offset) or (j%2 == 1 and i%2 == offset): # non green pixel
                out[j,i] = (raw[j+1,i] + raw[j-1,i] + raw[j,i+1]+raw[j,i-1])/4
            else:
                out[j,i] = raw[j,i]
    return out                       
                    
def basicRorB(raw, offsety, offsetx):
    h,w = raw.shape[0:2]
    out = numpy.zeros([h,w])
    for j in range(1, h-1):
        for i in range(1, w-1):
            if (j%2 == offsety and i%2 == offsetx): # non green pixel
                out[j,i] = raw[j,i]
            else:
                out[j,i] = (raw[j+1,i] + raw[j-1,i] + raw[j,i+1]+raw[j,i-1])/4
    return out        
  
def basicDemosaick(raw,offsetGreen=1, offsetRedY=1, offsetRedX=1, 
                   offsetBlueY=0, offsetBlueX=0): 
    g = basicGreen(raw, offsetGreen)
    r = basicRorB(raw, offsetRedY, offsetRedX)
    b = basicRorB(raw, offsetBlueY, offsetBlueX)
    im = numpy.array([r,g,b]).transpose(1,2,0)
    return im