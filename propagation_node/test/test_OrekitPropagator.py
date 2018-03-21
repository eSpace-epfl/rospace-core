import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...

from propagator.ThrustModel import *
from propagator.OrekitPropagator import *


class OrekitPropagatorTest(unittest.TestCase):

    def setUp(self):
        OrekitPropagator.init_jvm()

    def test_calculate_thrust(self):
        """Test that thrust changed correctly"""
        prop = OrekitPropagator()

        F_T = np.array([2.0, 1.4, 3.0])
        F_T_norm = np.linalg.norm(F_T, 2)
        Isp = float(1.0)

        prop.F_T = F_T
        prop.Isp = Isp
        prop.ThrustModel = ThrustModel()
        prop.calculate_thrust()

        self.assertTrue(prop.ThrustModel.firing)
        self.assertAlmostEqual(F_T_norm, prop.ThrustModel.getThrust())
        self.assertAlmostEqual(Isp, prop.ThrustModel.getISP())

    def test_zero_thrust(self):
        """Test that no thrust set if given Zero"""
        prop = OrekitPropagator()

        F_T = np.array([0.0, 0.0, 0.0])
        F_T_norm = np.linalg.norm(F_T, 2)
        Isp = float(1.0)

        prop.F_T = F_T
        prop.Isp = Isp
        prop.ThrustModel = ThrustModel()
        prop.calculate_thrust()

        self.assertFalse(prop.ThrustModel.firing)
        self.assertAlmostEqual(F_T_norm, prop.ThrustModel.getThrust())
        self.assertNotAlmostEqual(Isp, prop.ThrustModel.getISP())


if __name__ == '__main__':
    unittest.main()
