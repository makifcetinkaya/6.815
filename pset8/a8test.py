from a8 import *
from imageIO import *
from numpy import *
from a7help import *
import imageIO
import numpy as np
import unittest

class TestSequenceFunctions(unittest.TestCase):

#    def test0calculateLinBlendWeights(self):
#        im = imageIO.imread("stata-small-1.png")
#        ws = calcLinBlendWeights(im)
#        print ws
#        assert im.shape[0:2] == ws.shape[0:2]
#        
#        imwriteGrey(ws, 'stata-ws.png')
##                
#    def test1autostitch(self):
#        im1 = imageIO.imread("p1.png")
#        im2 = imageIO.imread("p2.png")
#        out = autostitch(im1,im2)
#        imageIO.imwrite(out, "room2.png")
    
    def test1autostitch(self):
        im1 = imageIO.imread("stata-1.png")
        im2 = imageIO.imread("stata-2.png")
        out = autostitch(im1,im2)
        imageIO.imwrite(out, "stata.png")

   
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)