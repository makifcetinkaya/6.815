from a2 import *
import imageIO
import numpy
import random
import unittest

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.im = imageIO.imread('bear.png')
#    def test_0_scaleNN(self):
#        out = scaleNN(self.im, 3.97)
#        imageIO.imwrite(out, 'test_scaleNN.png')
#    
#    def test_1_scaleLin(self):
#        out = scaleLin(self.im, .5)
#        imageIO.imwrite(out, 'test_scaleLin.png')
#    def test_3_getUV(self):
#        s = segment(0,0,0,10)
#        p = numpy.array([5,5])
#        assert getUV(s,p)[0] == .5
#        p = numpy. array([0, 10])
#        assert getUV(s,p)[0] == 1.0
#        s = segment(10,10, 10,20)
        #print getUV(s,p)
#        
#    def test_2_transform(self):
#        src_segment = segment(0, 0, 0, 10)  #x direction
#        dest_segment = segment(0, 0, 10, 0) #y dir
#        dest_point = numpy.array([10, 0])
#        src_point = transform(dest_point, dest_segment, src_segment)
#        assert src_point[0]== 0 and src_point[1]==10
#        src_segment = segment(10, 10, 10, 20)
#        dest_segment = segment(10, 10, 10, 30)
#        dest_point = numpy.array([10, 20])
#        src_point = transform(dest_point, dest_segment, src_segment)
#        print src_point
#        #assert src_point[0]== 10 and src_point[1]==20
##        
#
#    def test_2_warpBy1(self):
#        src_segment = segment(0, 0, 10, 0)  #x direction
#        dest_segment = segment(10, 10, 30, 15) #y direction
#        out = warpBy1(self.im, src_segment, dest_segment)
#        imageIO.imwrite(out, 'testWarp1.png')
        
#        
#    def test_4_warp(self):
#        im = imageIO.imread('fredo2.png')
#        listSegmentsBefore = numpy.array([segment(133, 122, 139, 167), segment(23, 109, 31, 159)])
#        listSegmentsAfter = numpy.array([segment(131, 121, 140, 169), segment(24, 108, 34, 158)])
#        out = warp(im, listSegmentsBefore, listSegmentsAfter)
#        imageIO.imwrite(out, 'fredo2t.png')
#    def test_5_morph(self):
#        im1 = imageIO.imread('fredo2.png')
#        im2 = imageIO.imread('werewolf.png')
#        listSegmentsBefore=numpy.array([segment(24, 108, 32, 157), segment(132, 121, 134, 165), segment(116, 187, 98, 197), segment(136, 190, 127, 218), segment(111, 220, 100, 203), segment(75, 112, 93, 101), segment(125, 109, 100, 100), segment(144, 113, 171, 109)])
#        listSegmentsAfter=numpy.array([segment(34, 102, 44, 143), segment(126, 102, 133, 132), segment(110, 153, 99, 171), segment(149, 148, 143, 188), segment(111, 192, 102, 178), segment(76, 102, 89, 94), segment(116, 100, 95, 91), segment(137, 94, 159, 85)])
#        morph(im1, im2, listSegmentsBefore, listSegmentsAfter, N=10)
#        
    def test_6_morph(self):
        im1 = imageIO.imread('class-4.png')
        im2 = imageIO.imread('class-5.png')
        listSegmentsBefore = numpy.array([segment(43, 119, 50, 167), segment(152, 123, 144, 170), segment(55, 175, 60, 196), segment(67, 203, 82, 215), segment(89, 217, 110, 217), segment(118, 214, 135, 198), segment(137, 190, 142, 176), segment(61, 109, 73, 99), segment(81, 100, 97, 108), segment(106, 108, 126, 98), segment(134, 100, 140, 108), segment(102, 114, 103, 152), segment(62, 118, 93, 120), segment(109, 121, 138, 120), segment(79, 172, 118, 173), segment(83, 178, 98, 182), segment(104, 183, 114, 178), segment(52, 125, 54, 92), segment(56, 82, 67, 57), segment(74, 59, 104, 63), segment(113, 66, 134, 71), segment(139, 79, 142, 99), segment(42, 108, 38, 71), segment(38, 61, 51, 29), segment(56, 19, 91, 4), segment(100, 0, 136, 19), segment(142, 25, 160, 58), segment(162, 66, 156, 115), segment(57, 203, 57, 215), segment(133, 207, 132, 217), segment(52, 224, 43, 239), segment(39, 245, 31, 255), segment(141, 218, 161, 243), segment(165, 249, 175, 256), segment(137, 232, 154, 257), segment(144, 230, 161, 255), segment(59, 229, 51, 237), segment(50, 244, 50, 255)])
        listSegmentsAfter = numpy.array([segment(38, 112, 48, 166), segment(152, 110, 142, 162), segment(57, 172, 66, 197), segment(70, 205, 84, 219), segment(91, 222, 106, 220), segment(115, 217, 130, 196), segment(135, 186, 139, 168), segment(60, 104, 69, 97), segment(77, 96, 94, 103), segment(103, 103, 123, 95), segment(131, 95, 138, 103), segment(99, 110, 100, 158), segment(62, 118, 90, 117), segment(109, 118, 135, 116), segment(79, 172, 119, 176), segment(83, 180, 100, 195), segment(108, 193, 117, 183), segment(51, 125, 54, 96), segment(58, 82, 62, 49), segment(71, 48, 95, 49), segment(107, 50, 124, 45), segment(132, 53, 140, 90), segment(41, 100, 42, 67), segment(41, 57, 55, 36), segment(61, 30, 87, 18), segment(95, 16, 127, 30), segment(132, 35, 142, 53), segment(146, 59, 150, 101), segment(60, 202, 61, 214), segment(133, 204, 132, 215), segment(56, 197, 29, 226), segment(22, 232, 3, 245), segment(140, 194, 172, 225), segment(179, 231, 198, 245), segment(139, 224, 133, 256), segment(147, 224, 160, 254), segment(57, 221, 49, 235), segment(50, 243, 50, 255)])
        morph(im1, im2, listSegmentsBefore, listSegmentsAfter, N=20)
        
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)    