from a7 import *
from imageIO import *
from numpy import *
import imageIO
import numpy as np
import unittest
from a7help import *

class TestSequenceFunctions(unittest.TestCase):

#    def test1computeTensor(self):
#        im = imageIO.imread("stata-1.png")
#        tensor = computeTensor(im)
#        imwrite(np.transpose(tensor,[1,2,0]), 'stata-t1.png')
#        im = imageIO.imread("stata-2.png")
#        tensor = computeTensor(im)
#        imwrite(np.transpose(tensor,[1,2,0]), 'stata-t2.png')
#                
#    def test2HarrisCorner(self):
#        im = imageIO.imread("stata-small-1.png")
#        cL = HarrisCorners(im)
#        visualizeCorners(im, cL)
#        
#        im = imageIO.imread("stata-small-2.png")
#        cL = HarrisCorners(im)
#        visualizeCorners(im, cL)
#        
#    def test3computeFeatures(self):
#        im = imageIO.imread("p1.png")
#        LF = computeFeatures(im)
#        visualizeFeatures(LF, 4, im)
#        
#        im = imageIO.imread("p2.png")
#        LF = computeFeatures(im)
#        visualizeFeatures(LF, 4, im)
#
#    def test5findCorrespondences(self):
#        im1 = imageIO.imread("stata-2.png")
#        LF1 = computeFeatures(im1)        
#        im2 = imageIO.imread("stata-1.png")
#        LF2 = computeFeatures(im2)
#        pairs = findCorrespondences(LF1, LF2)
#        visualizePairs(im1, im2, pairs)
#    def test6RANSAC(self):
#        im1 = imageIO.imread("stata-2.png")
#        LF1 = computeFeatures(im1)        
#        im2 = imageIO.imread("stata-1.png")
#        LF2 = computeFeatures(im2)
#        corrL = findCorrespondences(LF1, LF2)
#        
#        H, isInlierL = RANSAC(corrL, Niter=50, epsilon=4)
#        visualizePairsWithInliers(im1, im2, corrL, isInlierL)
#        
#    def test7autostitch(self):
#        im1 = imageIO.imread("stata-2.png")
#        im2 = imageIO.imread("stata-1.png")
#        out = autostitch(im1,im2)
#        imageIO.imwrite(out, "state-auto-st.png")
#    
#    def test8autostitch(self):
#        im1 = imageIO.imread("im4.png")
#        im2 = imageIO.imread("im5.png")
#        out = autostitch(im1,im2)
#        imageIO.imwrite(out, "room2.png")

    def test9autostitch(self):
        im1 = imageIO.imread("p1.png")
        im2 = imageIO.imread("p2.png")
        out = autostitch(im1,im2)
        imageIO.imwrite(out, "room2.png")
        
#    def test4findCorrespondence(self):
#        im1 = imageIO.imread("stata-1.png")
#        LF1 = computeFeatures(im1)
#        cornerL1, patchL1  = zip(*LF1)
#        patch2 = patchL1[0]
#        res = findCorrespondence(patch2, patchL1, cornerL1) 
#        assert res[1][0].all() == cornerL1[0].all()
#        
#        cornerL1, patchL1  = zip(*LF1)
#        patch2 = patchL1[5]
#        res = findCorrespondence(patch2, patchL1, cornerL1) 
#        assert res[1][0].all() == cornerL1[5].all()
        
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)