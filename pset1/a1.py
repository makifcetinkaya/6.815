import imageIO
import numpy

def brightness(im, factor):
    return im*factor

def contrast(im, factor, midpoint=.3):
    return im*factor+midpoint*(1-factor)

def frame(im2):
    im2 = im2.copy()
    im2[:,-1] = 0
    im2[:,0] = 0
    im2[1,:] = 0 
    im2[-1,:] = 0
    return im2

def dotInMiddle(im):
    (y,x) = im.shape[0:2]
    im[y/2,x/2,:] = [0,0,0]
    return im

def BW(im, weights=[.3,.6,.1]):
    bw = numpy.dot(im, weights)
    return numpy.array([bw,bw,bw]).transpose(1,2,0)
    

def lumiChromi(im):
    lum = BW(im)
    chrom = im/lum
    return (lum, chrom)

def brightnessContrastLumi(im, brightF, contrastF, midpoint=.3):
    (lum, chrom) = lumiChromi(im)  
    con = contrast(brightness(lum, brightF), contrastF, midpoint)
    return con*chrom

def rgb2yuv(im):
    conv = [[.299, .587, .114],
            [-.14713, -.28886, .436],
            [.615, -.51499, -.10001]]
    return numpy.dot(im, numpy.transpose(conv) )
    

def yuv2rgb(im):
    conv = [[1, 0, 1.13983],
            [1, -0.39465, -.58060],
            [1, 2.03211, 0]]
    return numpy.dot(im, numpy.transpose(conv))

def saturate(im, k):
    im_yuv = rgb2yuv(im)
    im_yuv[:,:,1:2] = im_yuv[:,:,1:2]*k
    return yuv2rgb(im_yuv)

def spanish(im):
    L = rgb2yuv(im)
    u = L[:,:,1]
    L[:,:,1] = L[:,:,2]
    L[:,:,2] = u
    dotInMiddle(L)
    imageIO.imwrite(L, 'L.png')
    C = BW(im)
    dotInMiddle(C)
    imageIO.imwrite(C, 'C.png')
   
def histogram(im, N):
    lum = BW(im)
    height, width = lum.shape[0:2]
    # the values will lie in the following range
    size = numpy.ceil(numpy.max(lum[:,:,0])*N)
    h = numpy.zeros(size)
    for i in range(height):
        for j in range(width):
            # size -> index: ceil -> floor ;)
            k = numpy.floor(lum[i,j,0]*N)
            h[k]= h[k] + 1  
    return h/(height*width)

def printHisto(im, N, scale):
    h = histogram(im, N)
    for k in range(h.shape[0]):
        print 'X'*int(h[k]*N*scale)
  
if __name__=="__main__":
    
    myarray = imageIO.imread('test.png')

   # myarray = brightness(myarray, 2)
   # myarray = frame(myarray)
    myarray = rgb2yuv(myarray)
    s = myarray.shape
    imageIO.imwrite(myarray, 'yuv3.png')
#    imageIO.imwrite(myarray[0], 'lum.png')
#    imageIO.imwrite(myarray[1], 'chrom.png')
#    imageIO.imwrite(myarray[1]*myarray[0], 'orig.png')
