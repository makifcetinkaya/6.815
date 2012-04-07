from a6 import *
from imageIO import *
from numpy import *
import imageIO
import numpy as np
import unittest

class TestSequenceFunctions(unittest.TestCase):
    
#    def test0applyhomography(self):
#        out = imageIO.imread('green.png')
#        source = imageIO.imread('poster.png')
#        H = np.array([[ 0.8346, -0.0058, -141.3292], [ 0.0116, 0.8025,-78.2148], [ -0.0002, -0.0006, 1. ]])
#        applyhomography(source, out, H, bilinear=True)
#        imageIO.imwrite(out, 'postered.png')
#        
#    def test1computehomography(self):
#        src = imageIO.imread('poster.png')
#        out = imread('green.png')
#        h, w = src.shape[0]-1, src.shape[1]-1
#        pointListPoster=[array([0, 0, 1]), array([0, w, 1]), array([h, w, 1]),array([h, 0, 1])]
#        pointListT=[array([170, 95, 1]), array([171, 238, 1]), array([233,235, 1]), array([239, 94, 1])]
#        listOfPairs =zip(pointListT, pointListPoster )
#        H=computehomography(listOfPairs)
#        applyhomography(src, out, H, True)
#        imwrite(out, 'test1.png')
#        
#    def test2computehomography(self):
#        im1=imread('stata-1.png')
#        im2=imread('stata-2.png')
#        pointList1=[array([209, 218, 1]), array([425, 300, 1]), array([209,337, 1]), array([396, 336, 1])]
#        pointList2=[array([232, 4, 1]), array([465, 62, 1]), array([247, 125, 1]), array([433, 102, 1])]
#        
#        listOfPairs=zip(pointList1, pointList2)
#        H=computehomography(listOfPairs)
#        out=im1*0.2
#        applyhomography(im2, out, H, True)
#        imwrite(out, 'test2.png')
#
#    def test3testcomputehomography_unit(self):
#        pointList1=[array([0, 0, 1]), array([0, 1, 1]), array([1, 0, 1]),array([1, 1, 1])]
#        pointList2=[array([0, 0, 1]), array([0, 2, 1]), array([2,0, 1]), array([2, 2, 1])]
#        # unity
#        listOfPairs = zip(pointList1, pointList1)
#        H = computehomography(listOfPairs)
#        print "unity: ", H    
#        
#    def test4computeTransformedBBox(self):
#        im1=imread('stata-1.png')
#        im2=imread('stata-2.png')
#        pointList1=[array([209, 218, 1]), array([425, 300, 1]), array([209,337, 1]), array([396, 336, 1])]
#        pointList2=[array([232, 4, 1]), array([465, 62, 1]), array([247, 125, 1]), array([433, 102, 1])]
#        
#        listOfPairs = zip(pointList1, pointList2)
#        H = computehomography(listOfPairs)
#        print H
#        print computeTransformedBBox(im1, H)
#     
#    def test5computeTranslatedBBox(self):
#        im = imread('stata-2.png')
#        pointList1=[array([209, 218, 1]), array([425, 300, 1]), array([209,337, 1]), array([396, 336, 1])]
#        pointList2=[array([232, 4, 1]), array([465, 62, 1]), array([247, 125, 1]), array([433, 102, 1])]
#        listOfPairs = zip(pointList1, pointList2)
#        H = computehomography(listOfPairs)
#        box = computeTransformedBBox(im, H)
#        print box
       
#       
#    def test6translate(self):
#       pointList1=[array([209, 218, 1]), array([425, 300, 1]), array([209,337, 1]), array([396, 336, 1])]
#       pointList2=[array([232, 4, 1]), array([465, 62, 1]), array([247, 125, 1]), array([433, 102, 1])]
#       listOfPairs = zip(pointList1, pointList2)
       

    def test8stitch(self):
        im1=imread('stata-small-1.png')
        im2=imread('stata-small-2.png')
        pointList1=[array([209, 218, 1]), array([425, 300, 1]), array([209,337, 1]), array([396, 336, 1])]
        pointList2=[array([232, 4, 1]), array([465, 62, 1]), array([247, 125, 1]), array([433, 102, 1])]
        
        listOfPairs = zip(pointList1, pointList2)
        out = stitch(im1, im2, listOfPairs)
        imwrite(out, "stata-stitched.png")
#    def test9(self):
#        im1 = imread('science-1.png') 
#        im2 = imread('science-2.png')
#        pointList1=[array([166, 76, 1]), array([306, 13, 1]), array([306, 105, 1]), array([154, 129, 1])]
#        pointList2=[array([159, 261, 1]), array([299, 211, 1]), array([293, 303, 1]), array([133, 316, 1])]
#        listOfPairs=zip(pointList2, pointList1)
#        out = stitch(im1, im2, listOfPairs)
#        imwrite(out, 'science-stitched.png')
        
         
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)