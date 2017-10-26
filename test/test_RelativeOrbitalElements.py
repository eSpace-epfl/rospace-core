# Note: these tests are quite preliminary....

import unittest
import sys
import os
from copy import deepcopy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../src/") #hack...
from space_tf import *

class RelativeOrbitalElementsTest(unittest.TestCase):

    def test_identity(self):
        target = OrbitalElements()
        target.e = 0.0007281
        target.i = 1.72
        target.w = 2.45
        target.t = 0.2
        target.omega = 1.39
        target.a = 7084

        chaser = target

        target_rel = QNSRelOrbElements()
        target_rel.from_absolute(target, chaser)

        # all relative elements should be 0 (same orbit)
        self.assertAlmostEqual(target_rel.dA, 0, places=10)
        self.assertAlmostEqual(target_rel.dL, 0, places=10)
        self.assertAlmostEqual(target_rel.dEx, 0, places=10)
        self.assertAlmostEqual(target_rel.dEy, 0, places=10)
        self.assertAlmostEqual(target_rel.dIx, 0, places=10)
        self.assertAlmostEqual(target_rel.dIy, 0, places=10)

        # convert back
        target_2 = target_rel.to_absolute(chaser)
        self.assertAlmostEqual(target.a, target_2.a, places=10)
        self.assertAlmostEqual(target.e, target_2.e, places=10)
        self.assertAlmostEqual(target.i, target_2.i, places=10)
        self.assertAlmostEqual(target.omega, target_2.omega, places=10)
        self.assertAlmostEqual(target.w, target_2.w, places=10)
        self.assertAlmostEqual(target.m, target_2.m, places=10)
        self.assertAlmostEqual(target.t, target_2.t, places=10)

    def test_deltaL(self):
        target = OrbitalElements()
        target.e = 0.0007281
        target.i = 1.72
        target.w = 2.45
        target.t = 0.2
        target.omega = 1.39
        target.a = 7084

        chaser = deepcopy(target)
        chaser.a = 7085
        chaser.t = 0.1

        target_rel = QNSRelOrbElements()
        target_rel.from_absolute(target, chaser)

        # all relative elements should be 0 (same orbit)
        # todo: get testcase where ROE are known
        self.assertNotAlmostEqual(target_rel.dA, 0, places=10)
        self.assertNotAlmostEqual(target_rel.dL, 0, places=10)
        self.assertAlmostEqual(target_rel.dEx, 0, places=10)
        self.assertAlmostEqual(target_rel.dEy, 0, places=10)
        self.assertAlmostEqual(target_rel.dIx, 0, places=10)
        self.assertAlmostEqual(target_rel.dIy, 0, places=10)

        # convert back
        target_2 = target_rel.to_absolute(chaser)

        self.assertAlmostEqual(target.a, target_2.a, places=10)
        self.assertAlmostEqual(target.e, target_2.e, places=10)
        self.assertAlmostEqual(target.i, target_2.i, places=10)
        self.assertAlmostEqual(target.omega, target_2.omega, places=10)
        self.assertAlmostEqual(target.w, target_2.w, places=10)
        self.assertAlmostEqual(target.m, target_2.m, places=10)
        self.assertAlmostEqual(target.t, target_2.t, places=10)


if __name__ == '__main__':
    unittest.main()
