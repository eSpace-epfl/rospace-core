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
        v_test = 0.787398241564565646
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


if __name__ == '__main__':
    unittest.main()
