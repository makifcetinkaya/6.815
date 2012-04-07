import imageIO
import numpy
import random
import unittest
from a5 import *

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        pass
    
#    def test_0_computeWeight(self):
#        im1 = imageIO.imread('ante1-1.png')
#        im2 = imageIO.imread('ante1-2.png')
#        w1 = computeWeight(im1)
#        w2 = computeWeight(im2)
#        imageIO.imwrite(w1, 'ante11w.png')
#        imageIO.imwrite(w2, 'ante12w.png')
        
#    def test_1_computeFactor(self):
#        im1 = imageIO.imread('ante1-1.png')
#        im2 = imageIO.imread('ante1-2.png')
#        w1 = computeWeight(im1)
#        w2 = computeWeight(im2)
#        
#        factor = computeFactor(im1, w1, im2, w2)
#        print factor
#        
#    def test_2_makeHDR(self):
#        k = 1
#        images = []
#        for i in range(6):
#            images.append(imageIO.imread('design-'+str(i+1)+'.png'))
#        hdr = makeHDR(images)
#        imageIO.imwrite(hdr*k, 'design-hdr-k'+str(k)+'.png')
##        numpy.save('design-hdr.npy', hdr)
#        
#        images=[]
#        for i in range(3):
#            images.append(imageIO.imread('vine-'+str(i+1)+'.png'))
#        hdr = makeHDR(images)
#        imageIO.imwrite(hdr*k, 'vine-hdr-k'+str(k)+'.png')
##        numpy.save('vine-hdr.npy', hdr)
#        
#        images=[]
#        for i in range(2):
#            images.append(imageIO.imread('ante2-'+str(i+1)+'.png'))
#        hdr = makeHDR(images)
#        imageIO.imwrite(hdr*k, 'ante2-hdr-k'+str(k)+'.png')
#        numpy.save('ante2-hdr.npy', hdr)
        
    def test_3_toneMap(self):
        im1 = numpy.load('vine-hdr.npy')
#        im2 = numpy.load('design-hdr.npy')
#        im3 = numpy.load('ante2-hdr.npy')
        t1 = toneMap(im1, 100, 1, False)
#        t2 = toneMap(im2)
#        t3 = toneMap(im3)
        
        imageIO.imwrite(t1, 'vine-toned-g.png')
#        imageIO.imwrite(t2, 'design-toned.png')
#        imageIO.imwrite(t3, 'ante2-toned.png')
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
