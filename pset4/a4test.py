import imageIO
import numpy
import random
import unittest
from a4 import *

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
#        self.imageList3200, self.imageList400 = [],[]
#        for i in range(6):
#            path = '1D2N-iso3200-'+str(i+1)+'.png'
#            im = imageIO.imread(path)
#            self.imageList3200.append(im)
#            
#            path = '1D2N-iso400-under-'+str(i+1)+'.png'
#            im = imageIO.imread(path)
#            self.imageList400.append(im)
        
#        self.noiseImageList = []
#        for i in range(3):
#            path = 'noise-small-'+str(i+1)+'.png'
#            im = imageIO.imread(path)
#            self.noiseImageList.append(im)
        pass
        #self.im = imageIO.imread('test.png')

#    def test_0_denoiseSeq(self):
#        imageIO.imwrite(denoiseSeq(self.imageList),'avr.png')
#        
#    def test_1_logSNR(self):
#        snr3200 = logSNR(self.imageList3200)
#        snr400 = logSNR(self.imageList400)
#        imageIO.imwrite(snr3200, 'snr32.png')
#        imageIO.imwrite(snr400,'snr4.png')
        
#    def test_2_align(self):
#        im1 = imageIO.constantIm(50,50,[.5,0.5,0.5])
#        im2 = numpy.copy(im1)
#        im2[21,29] = [.3,.8,.7]
#        im1[25,25] = [1,1,1]
#        
#        indices = align(im1, im2, maxOffset=5)
#        assert indices == (4,-4)

#    def test_5_grey(self):
#        im = imageIO.imreadGrey('signs-small.png')
#    def test_6_align(self):
#        im1 = self.noiseImageList[0]
#        im2 = self.noiseImageList[1]
#        shift = align(im1,im2,10)
#        imageIO.imwrite(numpy.roll(numpy.roll(im2, shift[0], 0),shift[1],1),'shifted1.png')
#        
#    def test_8_alignAndDenoise(self):
#        a_n = alignAndDenoise(self.noiseImageList, 20)
#        imageIO.imwrite(a_n, 'aligned_denoised.png')
#    
#    def test_4_denoise(self):
#        imageIO.imwrite(denoiseSeq(self.noiseImageList), 'just_denoised.png')

#    def test_9_basicGreen(self):
#        raw = imageIO.imreadGrey('signs-small.png')
#        green = basicGreen(raw)
#        im = numpy.array([green,green,green]).transpose(1,2,0)
#        imageIO.imwrite(im, 'signs-small-BG.png')
    def test_10_basicDemosaick(self):
        raw = imageIO.imreadGrey('signs-small.png')
        im = basicDemosaick(raw)
        imageIO.imwrite(im, 'signs-small-basic.png')
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
