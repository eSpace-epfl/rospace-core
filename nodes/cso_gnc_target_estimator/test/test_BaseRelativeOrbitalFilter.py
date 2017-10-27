# Note: these tests are quite preliminary....

import numpy as np
import os
import rospy
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../src/")  # hack...
from BaseRelativeOrbitalFilter import BaseRelativeOrbitalFilter


class BaseRelativOrbitalFilterTest(unittest.TestCase):

    def test_H(self):
        f = BaseRelativeOrbitalFilter()
        f.x = np.array([[4.60056745e-04, 1.87914228e-03, -2.68793742e-06,  -1.91226006e-06, -1.34998767e-07,  -4.42746334e-04]]).T
        f.T_c = 3.45093781738
        f.alpha_c = np.array([[7.06904991e+03, 3.45100760e+00, 2.49896846e-03, 1.36546097e+00, 1.71935190e+00, 6.76581678e-01]]).T
        f.time_c = rospy.Time(1, 17568000)
        f.R_body = np.zeros([4,4])
        f.R_body[0,:] = [-0.0110633, 0.78539025, 0.61890206,  0.]
        f.R_body[1,:] = [ 0.1801984, 0.61037276, -0.77134534, 0.]
        f.R_body[2,:] = [-0.98356807, 0.10299153, -0.1482788, 0.]
        f.R_body[3,:] = [0, 0, 0, 1]


        for i in range(0,6):
            x_1 = f.x.copy()
            y_1,foo = f.get_H_of_x()
            H = f.get_H()
            f.x[i] = f.x[i]+f.x[i]*0.01
            x_2 = f.x.copy()
            y_2,foo = f.get_H_of_x()

            print H.dot(x_2-x_1).T
            print (y_2-y_1).T
            print np.linalg.norm(H.dot(x_2-x_1)-(y_2-y_1))/np.linalg.norm((y_2-y_1))
            print "===="

if __name__ == '__main__':
    unittest.main()
