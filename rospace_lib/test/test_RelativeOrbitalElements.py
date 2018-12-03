# Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.

# Note: these tests are quite preliminary....

import unittest
import sys
import os
from copy import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
from rospace_lib import *


class RelativeOrbitalElementsTest(unittest.TestCase):
    def test_identity(self):
        target = KepOrbElem()
        target.e = 0.0007281
        target.i = 1.72
        target.w = 2.45
        target.v = 0.2
        target.O = 1.39
        target.a = 7084

        chaser = target

        target_rel = QNSRelOrbElements()
        target_rel.from_keporb(target, chaser)

        # all relative elements should be 0 (same orbit)
        self.assertAlmostEqual(target_rel.dA, 0, places=10)
        self.assertAlmostEqual(target_rel.dL, 0, places=10)
        self.assertAlmostEqual(target_rel.dEx, 0, places=10)
        self.assertAlmostEqual(target_rel.dEy, 0, places=10)
        self.assertAlmostEqual(target_rel.dIx, 0, places=10)
        self.assertAlmostEqual(target_rel.dIy, 0, places=10)

        # convert back
        target_2 = KepOrbElem()
        target_2.from_qns_relative(target_rel, chaser)
        self.assertAlmostEqual(target.a, target_2.a, places=10)
        self.assertAlmostEqual(target.e, target_2.e, places=10)
        self.assertAlmostEqual(target.i, target_2.i, places=10)
        self.assertAlmostEqual(target.O, target_2.O, places=10)
        self.assertAlmostEqual(target.w, target_2.w, places=10)
        self.assertAlmostEqual(target.m, target_2.m, places=10)
        self.assertAlmostEqual(target.v, target_2.v, places=10)
        self.assertAlmostEqual(target.E, target_2.E, places=10)

    def test_deltaL(self):
        target = KepOrbElem()
        target.e = 0.0007281
        target.i = 1.72
        target.w = 2.45
        target.v = 0.2
        target.omega = 1.39
        target.a = 7084

        chaser = copy(target)
        chaser.a = 7085
        chaser.v = 0.1

        target_rel = QNSRelOrbElements()
        target_rel.from_keporb(target, chaser)

        # dA an dL are different, rest same
        # todo: get testcase where ROE are known
        self.assertNotAlmostEqual(target_rel.dA, 0, places=10)
        self.assertNotAlmostEqual(target_rel.dL, 0, places=10)
        self.assertAlmostEqual(target_rel.dEx, 0, places=10)
        self.assertAlmostEqual(target_rel.dEy, 0, places=10)
        self.assertAlmostEqual(target_rel.dIx, 0, places=10)
        self.assertAlmostEqual(target_rel.dIy, 0, places=10)

        # convert back
        target_2 = KepOrbElem()

        target_2.from_qns_relative(target_rel, chaser)

        self.assertAlmostEqual(target.a, target_2.a, places=10)
        self.assertAlmostEqual(target.e, target_2.e, places=10)
        self.assertAlmostEqual(target.i, target_2.i, places=10)
        self.assertAlmostEqual(target.O, target_2.O, places=10)
        self.assertAlmostEqual(target.w, target_2.w, places=10)
        self.assertAlmostEqual(target.m, target_2.m, places=10)
        self.assertAlmostEqual(target.v, target_2.v, places=10)
        self.assertAlmostEqual(target.E, target_2.E, places=10)


if __name__ == '__main__':
    unittest.main()
