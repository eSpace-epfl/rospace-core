# Note: these tests are quite preliminary....

import unittest
import sys
import os
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
from space_tf import *


class KepOrbElemTest(unittest.TestCase):
    def test_true_to_mean_anomaly(self):

        for i in np.arange(0.1,np.pi*2, 0.1):
            v_test = i
            source = KepOrbElem()
            source.e = 0.01
            source.a = 7000
            source.i = 0.1
            source.w = 0.1
            source.O = 0.2
            source.v = v_test

            mean = source.m

            # reset v
            source.v = 0.
            self.assertNotAlmostEqual(v_test, source.v)

            source.m = mean

            self.assertAlmostEqual(v_test, source.v, places=10)

    def test_mean_to_true_anomaly(self):

        for i in np.arange(0.1, np.pi * 2, 0.1):
            m_test = i
            source = KepOrbElem()
            source.e = 0.01
            source.a = 7000
            source.i = 0.1
            source.w = 0.1
            source.O = 0.2
            source.m = m_test

            v_calc = source.v

            # reset m
            source.m = 0.
            self.assertNotAlmostEqual(m_test, source.m)

            source.v = v_calc

            self.assertAlmostEqual(m_test, source.m, places=10)

    def test_true_to_ecc_anomaly(self):
        v_test = 0.787398241564565646
        source = KepOrbElem()
        source.e = 0.01
        source.a = 7000
        source.i = 0.1
        source.w = 0.1
        source.O = 0.2
        source.v = v_test

        ecc = source.E

        # reset v
        source.v = 0.
        self.assertNotAlmostEqual(v_test, source.v)

        source.E = ecc

        self.assertAlmostEqual(v_test, source.v, places=10)

    def test_mean_to_ecc_anomaly(self):
        m_test = 0.787398241564565646
        source = KepOrbElem()
        source.e = 0.01
        source.a = 7000
        source.i = 0.1
        source.w = 0.1
        source.O = 0.2
        source.m = m_test

        ecc = source.E

        # reset m
        source.m = 0.
        self.assertNotAlmostEqual(m_test, source.m)

        source.E = ecc

        self.assertAlmostEqual(m_test, source.m, places=10)

    def test_occ_elems(self):
        source = OscKepOrbElem()
        source.e = 0.000891342369007
        source.a = 7088.68813833
        source.i = np.deg2rad(98.5006834885)
        source.w = np.deg2rad(69.2680911804)
        source.O = np.deg2rad(38.7326244021)
        source.v = np.deg2rad(100.250407006)

        target = KepOrbElem()
        target.from_osc_elems(source)


        target2 = OscKepOrbElem()
        target2.from_mean_elems(target)

        print "e", source.e - target2.e
        print "w", source.w - target2.w
        print "O", source.O - target2.O
        print "m", source.m - target2.m
        print "v", source.v - target2.v
        print "i", source.i - target2.i
        print "a", source.a - target2.a

        print "GUGUS"


if __name__ == '__main__':
    unittest.main()
