# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import unittest
import sys
import os
import numpy as np
from datetime import datetime 
from itertools import izip
import math

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...

from propagator.OrekitPropagator import *
from propagator.DipoleModel import *
from propagator.AttitudePropagation import *

# add test which checks rotation of magnetic field is correct with numpy rot matrix.
# so far only test needed to validate attitude prop. ---------------------


class MagneticTorqueTest(unittest.TestCase):
    dipolM = None

    def setUp(self):
        OrekitPropagator.init_jvm()

        # get orekit files for magnetic field
        # self.epoch = datetime(2011, 02, 11)
        # OrekitPropagator.load_magnetic_field_models(self.epoch)

    @classmethod
    def setUpClass(cls):
        MagneticTorqueTest.dipolM = DipoleModel()

        Br = [0.35, 0.005, 0.11]
        Bs = [0.74, 0.025, 0.42]
        Hc = [0.96, 10., 0.85]
        dir_hyst = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
        volume = [7.4613e-8, 2.1234e-8, 1.178e-7]

        bar_mag_m = [0.3, 0.43]
        dir_bar = np.array([[1, 0, 0], [1, 0, 1]])

        for r, s, c, direc, vol in izip(Br, Bs, Hc, dir_hyst, volume):
            MagneticTorqueTest.dipolM.addHysteresis(direc, vol, c, s, Br=r)

        for m, direc in izip(bar_mag_m, dir_bar):
            MagneticTorqueTest.dipolM.addBarMagnet(direc, m)

    def test_add_material(self):
        self.assertAlmostEquals(MagneticTorqueTest.dipolM._k[0], 0.956774, delta=1e-6)
        self.assertAlmostEquals(MagneticTorqueTest.dipolM._k[1], 0.032492, delta=1e-6)
        self.assertAlmostEquals(MagneticTorqueTest.dipolM._k[2], 0.513289, delta=1e-6)

        self.assertEqual(MagneticTorqueTest.dipolM._H_hyst_last.size, 3)
        self.assertEqual(MagneticTorqueTest.dipolM._B_hyst_last.size, 3)
        self.assertEqual(MagneticTorqueTest.dipolM._Hc.size, 3)
        self.assertEqual(MagneticTorqueTest.dipolM._Bs.size, 3)
        self.assertEqual(MagneticTorqueTest.dipolM._m_barMag.size, 6)

    def test_initialization(self):
        B_field = np.array([1.5e-5, -2.2e-4, 0.5e-7])

        MagneticTorqueTest.dipolM.initializeHysteresisModel(B_field)

        self.assertAlmostEquals(MagneticTorqueTest.dipolM._H_hyst_last[0], 0.5e-7 / (math.pi * 4e-7), delta=1e-3)
        self.assertAlmostEquals(MagneticTorqueTest.dipolM._H_hyst_last[1],  -0.000144957 / (math.pi * 4e-7), delta=1e-3)
        self.assertAlmostEquals(MagneticTorqueTest.dipolM._H_hyst_last[2], -2.2e-4 / (math.pi * 4e-7), delta=1e-3)

        self.assertAlmostEquals(MagneticTorqueTest.dipolM._B_hyst_last[0], -0.340086, delta=1e-6)
        self.assertAlmostEquals(MagneticTorqueTest.dipolM._B_hyst_last[1], -0.020477, delta=1e-6)
        self.assertAlmostEquals(MagneticTorqueTest.dipolM._B_hyst_last[2], -0.417010, delta=1e-6)

    def test_dipole_calculation(self):
        B_field = np.array([1.5e-5, -2.2e-4, 0.5e-7])

        dipoleVectors = MagneticTorqueTest.dipolM.getDipoleVectors(B_field)

        self.assertAlmostEquals(dipoleVectors[0, 0], 0., delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[0, 1], 0., delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[0, 2], -0.340086 * 7.4613e-8 / (math.pi * 4e-7), delta=1e-6)

        self.assertAlmostEquals(dipoleVectors[1, 0], -0.020477 * 2.1234e-8 / (math.pi * 4e-7 * math.sqrt(2)), delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[1, 1], -0.020477 * 2.1234e-8 / (math.pi * 4e-7 * math.sqrt(2)), delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[1, 2], 0., delta=1e-6)

        self.assertAlmostEquals(dipoleVectors[2, 0], 0., delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[2, 1], -0.417010 * 1.178e-7 / (math.pi * 4e-7), delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[2, 2], 0., delta=1e-6)

        self.assertAlmostEquals(dipoleVectors[3, 0], 0.3, delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[3, 1], 0., delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[3, 2], 0., delta=1e-6)

        self.assertAlmostEquals(dipoleVectors[4, 0], 0.43 / math.sqrt(2), delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[4, 1], 0., delta=1e-6)
        self.assertAlmostEquals(dipoleVectors[4, 2], 0.43 / math.sqrt(2), delta=1e-6)

    def test_nothing_added(self):
        dipolM_nothing = DipoleModel()
        B_field = np.array([1.5e-5, -2.2e-4, 0.5e-7])

        dipoleVectors = dipolM_nothing.getDipoleVectors(B_field)

        self.assertEqual(dipoleVectors.size, 0)
