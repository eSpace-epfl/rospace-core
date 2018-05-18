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

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...

from propagator.OrekitPropagator import *
from propagator.SatelliteDiscretization import *


class SatelliteDiscretizationTest(unittest.TestCase):

    def setUp(self):
        OrekitPropagator.init_jvm()

        self.satDisc = BoxWingModel()

    @staticmethod
    def create_satellite_model(booms=True, s_arrays=True):

        disc_s = dict()
        disc_s['satellite_dim'] = dict()
        disc_s['inner_cuboids'] = dict()
        disc_s['surface_rectangles'] = dict()

        disc_s['satellite_dim']['l_x'] = 2.8
        disc_s['satellite_dim']['l_y'] = 1.2
        disc_s['satellite_dim']['l_z'] = 2.

        disc_s['inner_cuboids']['numCub_x'] = 4
        disc_s['inner_cuboids']['numCub_y'] = 2
        disc_s['inner_cuboids']['numCub_z'] = 3

        disc_s['surface_rectangles']['numSR_x'] = 3
        disc_s['surface_rectangles']['numSR_y'] = 4
        disc_s['surface_rectangles']['numSR_z'] = 2

        if booms:
            disc_s['Booms'] = dict()
            disc_s['Booms']['B1'] = dict()
            disc_s['Booms']['B1']['length'] = 20
            disc_s['Booms']['B1']['dir'] = '1 0 1'
            disc_s['Booms']['B1']['mass'] = 123

        if s_arrays:
            disc_s['SolarArrays'] = dict()
            disc_s['SolarArrays']['PosAndDir'] = dict()
            disc_s['SolarArrays']['PosAndDir']['SA1'] = dict()
            disc_s['SolarArrays']['PosAndDir']['l_x'] = 1.5
            disc_s['SolarArrays']['PosAndDir']['l_z'] = 0.5
            disc_s['SolarArrays']['PosAndDir']['numSRSolar_x'] = 2
            disc_s['SolarArrays']['PosAndDir']['numSRSolar_z'] = 3
            disc_s['SolarArrays']['PosAndDir']['SA1']['dCenter'] = '0 1 0'
            disc_s['SolarArrays']['PosAndDir']['SA1']['normalV'] = '0 0 1'

        solar_s = dict()
        solar_s['AbsorbCoeff'] = 0.8
        solar_s['ReflectCoeff'] = 0.2
        solar_s['SolarArray_AbsorbCoeff'] = 0.92
        solar_s['SolarArray_ReflectCoeff'] = 0.08

        return [disc_s, solar_s]

    def test_inner_cuboid_discretization(self):
        '''Test that inner satellite is discretized correctly.

        According to settings in @create_satellite_model following coordinates should be
        output by discretization:
            x-Dir: [-0.35-0.7 ; -0.35 ; 0.35 ; 0.35 + 0.7]
            y-Dir: [-0.3 ; 0.3]
            z-Dir: [-0.66 ; 0 ; 0.66]

        And total 24 cuboids provided

        No booms are being added to discretization and solar arrays should not be discretized
        as assumed 2D.
        '''
        disc_s, _ = self.create_satellite_model(booms=False, s_arrays=False)
        inCub = self.satDisc.discretize_inner_body(disc_s)

        com = inCub['CoM_np']
        dm = inCub['dm']
        # uniform mass distribution
        for mass_fraction in dm:
            self.assertAlmostEqual(mass_fraction, 1. / 24., delta=1e-6)

        # first 3 cuboids furthest away in negative direction, bottom to top
        # 0th step in x-direction
        self.assertAlmostEqual(com[0][0], -0.35 - 0.7, delta=1e-6)
        self.assertAlmostEqual(com[0][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[0][2], - 2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[1][0], -0.35 - 0.7, delta=1e-6)
        self.assertAlmostEqual(com[1][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[1][2], - 0,  delta=1e-6)
        self.assertAlmostEqual(com[2][0], -0.35 - 0.7, delta=1e-6)
        self.assertAlmostEqual(com[2][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[2][2],  2. / 3.,  delta=1e-6)
        # next 3 cuboids (shift in y direction)
        self.assertAlmostEqual(com[3][0], -0.35 - 0.7, delta=1e-6)
        self.assertAlmostEqual(com[3][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[3][2], - 2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[4][0], -0.35 - 0.7, delta=1e-6)
        self.assertAlmostEqual(com[4][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[4][2], - 0,  delta=1e-6)
        self.assertAlmostEqual(com[5][0], -0.35 - 0.7, delta=1e-6)
        self.assertAlmostEqual(com[5][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[5][2],  2. / 3.,  delta=1e-6)

        #next 6 cuboids, 1st step in positive x-direction
        self.assertAlmostEqual(com[6][0], -0.35, delta=1e-6)
        self.assertAlmostEqual(com[6][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[6][2], - 2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[7][0], -0.35, delta=1e-6)
        self.assertAlmostEqual(com[7][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[7][2], - 0,  delta=1e-6)
        self.assertAlmostEqual(com[8][0], -0.35, delta=1e-6)
        self.assertAlmostEqual(com[8][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[8][2],  2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[9][0], -0.35, delta=1e-6)
        self.assertAlmostEqual(com[9][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[9][2], - 2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[10][0], -0.35, delta=1e-6)
        self.assertAlmostEqual(com[10][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[10][2], - 0,  delta=1e-6)
        self.assertAlmostEqual(com[11][0], -0.35, delta=1e-6)
        self.assertAlmostEqual(com[11][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[11][2],  2. / 3.,  delta=1e-6)

        # next 6 cuboids, 2nd step in positive x-direction
        self.assertAlmostEqual(com[12][0], 0.35, delta=1e-6)
        self.assertAlmostEqual(com[12][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[12][2], - 2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[13][0], 0.35, delta=1e-6)
        self.assertAlmostEqual(com[13][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[13][2], - 0,  delta=1e-6)
        self.assertAlmostEqual(com[14][0], 0.35, delta=1e-6)
        self.assertAlmostEqual(com[14][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[14][2],  2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[15][0], 0.35, delta=1e-6)
        self.assertAlmostEqual(com[15][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[15][2], - 2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[16][0], 0.35, delta=1e-6)
        self.assertAlmostEqual(com[16][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[16][2], - 0,  delta=1e-6)
        self.assertAlmostEqual(com[17][0], 0.35, delta=1e-6)
        self.assertAlmostEqual(com[17][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[17][2],  2. / 3.,  delta=1e-6)

        # next 6 cuboids, 3nd step in positive x-direction
        self.assertAlmostEqual(com[18][0], 0.35 + 0.7, delta=1e-6)
        self.assertAlmostEqual(com[18][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[18][2], - 2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[19][0], 0.35 + 0.7, delta=1e-6)
        self.assertAlmostEqual(com[19][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[19][2], - 0,  delta=1e-6)
        self.assertAlmostEqual(com[20][0], 0.35 + 0.7, delta=1e-6)
        self.assertAlmostEqual(com[20][1], -0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[20][2],  2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[21][0], 0.35 + 0.7, delta=1e-6)
        self.assertAlmostEqual(com[21][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[21][2], - 2. / 3.,  delta=1e-6)
        self.assertAlmostEqual(com[22][0], 0.35 + 0.7, delta=1e-6)
        self.assertAlmostEqual(com[22][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[22][2], - 0,  delta=1e-6)
        self.assertAlmostEqual(com[23][0], 0.35 + 0.7, delta=1e-6)
        self.assertAlmostEqual(com[23][1], 0.6 * 0.5, delta=1e-6)
        self.assertAlmostEqual(com[23][2],  2. / 3.,  delta=1e-6)

    def test_boom_discretization(self):
        """Test that the booms/point masses are added correctly to arrays

        Booms should be added at end of array.
        """
        disc_s, _ = self.create_satellite_model()
        inCub = self.satDisc.discretize_inner_body(disc_s)

        com = inCub['CoM_np']
        com_boom = com[24]

        dm = inCub['dm_boom']

        self.assertAlmostEqual(com_boom[0], 0.707106781 * 20, delta=1e-6)
        self.assertAlmostEqual(com_boom[1], 0, delta=1e-6)
        self.assertAlmostEqual(com_boom[2], 0.707106781 * 20, delta=1e-6)

        self.assertEqual(dm[0], 123)

    def test_outer_surface_discretization(self):
        '''Test that outher surface discretized correctly.
        Only front and back facets are tested
        According to settings in @create_satellite_model following coordinates should be
        output by discretization for front and back facet:
            x-Dir: [1.4 ; -1.4]
            y-Dir: [-0.45, -0.15, 0.15, 0.45]
            z-Dir: [-0.5 ; 0.5]

        And total 24 cuboids provided

        No booms are being added to discretization and solar arrays should not be discretized
        as assumed 2D.
        '''
        disc_s, solar_s = self.create_satellite_model(booms=False, s_arrays=False)
        mesh_dA = self.satDisc.discretize_outer_surface(solar_s, disc_s)

        com = mesh_dA['CoM_np']

        print com
        # always front and back side
        self.assertAlmostEqual(com[0][0], 1.4, delta=1e-6)
        self.assertAlmostEqual(com[0][1], -(0.3 + 0.15), delta=1e-6)
        self.assertAlmostEqual(com[0][2], -0.5, delta=1e-6)
        self.assertAlmostEqual(com[1][0], -1.4, delta=1e-6)
        self.assertAlmostEqual(com[1][1], -(0.3 + 0.15), delta=1e-6)
        self.assertAlmostEqual(com[1][2], -0.5, delta=1e-6)

        self.assertAlmostEqual(com[2][0], 1.4, delta=1e-6)
        self.assertAlmostEqual(com[2][1], -(0.3 + 0.15), delta=1e-6)
        self.assertAlmostEqual(com[2][2], 0.5, delta=1e-6)
        self.assertAlmostEqual(com[3][0], -1.4, delta=1e-6)
        self.assertAlmostEqual(com[3][1], -(0.3 + 0.15), delta=1e-6)
        self.assertAlmostEqual(com[3][2], 0.5, delta=1e-6)

        self.assertAlmostEqual(com[4][0], 1.4, delta=1e-6)
        self.assertAlmostEqual(com[4][1], -(0.15), delta=1e-6)
        self.assertAlmostEqual(com[4][2], -0.5, delta=1e-6)
        self.assertAlmostEqual(com[5][0], -1.4, delta=1e-6)
        self.assertAlmostEqual(com[5][1], -(0.15), delta=1e-6)
        self.assertAlmostEqual(com[5][2], -0.5, delta=1e-6)

        self.assertAlmostEqual(com[6][0], 1.4, delta=1e-6)
        self.assertAlmostEqual(com[6][1], -(0.15), delta=1e-6)
        self.assertAlmostEqual(com[6][2], 0.5, delta=1e-6)
        self.assertAlmostEqual(com[7][0], -1.4, delta=1e-6)
        self.assertAlmostEqual(com[7][1], -(0.15), delta=1e-6)
        self.assertAlmostEqual(com[7][2], 0.5, delta=1e-6)

        self.assertAlmostEqual(com[8][0], 1.4, delta=1e-6)
        self.assertAlmostEqual(com[8][1], 0.15, delta=1e-6)
        self.assertAlmostEqual(com[8][2], -0.5, delta=1e-6)
        self.assertAlmostEqual(com[9][0], -1.4, delta=1e-6)
        self.assertAlmostEqual(com[9][1], 0.15, delta=1e-6)
        self.assertAlmostEqual(com[9][2], -0.5, delta=1e-6)

        self.assertAlmostEqual(com[10][0], 1.4, delta=1e-6)
        self.assertAlmostEqual(com[10][1], 0.15, delta=1e-6)
        self.assertAlmostEqual(com[10][2], 0.5, delta=1e-6)
        self.assertAlmostEqual(com[11][0], -1.4, delta=1e-6)
        self.assertAlmostEqual(com[11][1], 0.15, delta=1e-6)
        self.assertAlmostEqual(com[11][2], 0.5, delta=1e-6)

        self.assertAlmostEqual(com[12][0], 1.4, delta=1e-6)
        self.assertAlmostEqual(com[12][1], 0.3 + 0.15, delta=1e-6)
        self.assertAlmostEqual(com[12][2], -0.5, delta=1e-6)
        self.assertAlmostEqual(com[13][0], -1.4, delta=1e-6)
        self.assertAlmostEqual(com[13][1], 0.3 + 0.15, delta=1e-6)
        self.assertAlmostEqual(com[13][2], -0.5, delta=1e-6)

        self.assertAlmostEqual(com[14][0], 1.4, delta=1e-6)
        self.assertAlmostEqual(com[14][1], 0.3 + 0.15, delta=1e-6)
        self.assertAlmostEqual(com[14][2], 0.5, delta=1e-6)
        self.assertAlmostEqual(com[15][0], -1.4, delta=1e-6)
        self.assertAlmostEqual(com[15][1], 0.3 + 0.15, delta=1e-6)
        self.assertAlmostEqual(com[15][2], 0.5, delta=1e-6)

    def test_impossible_length(self):
        """Test that negative edge lenght results in error and not explosion."""
        disc_s, solar_s = self.create_satellite_model(booms=True, s_arrays=False)

        disc_s['satellite_dim']['l_x'] = -2.8
        self.assertRaises(ValueError, self.satDisc.discretize_inner_body, disc_s)

        self.assertRaises(ValueError, self.satDisc.discretize_outer_surface, solar_s, disc_s)

    def test_no_cuboids(self):
        """Test that 0 cuboids for discretization results in error."""
        disc_s, _ = self.create_satellite_model(booms=False, s_arrays=False)

        disc_s['inner_cuboids']['numCub_y'] = 0
        self.assertRaises(ValueError, self.satDisc.discretize_inner_body, disc_s)

        disc_s['inner_cuboids']['numCub_y'] = 1
        disc_s['inner_cuboids']['numCub_z'] = -2
        self.assertRaises(ValueError, self.satDisc.discretize_inner_body, disc_s)

    def test_no_surface_number(self):
        """Test that 0 surface planes for discretization results in error."""
        disc_s, solar_s = self.create_satellite_model(booms=False, s_arrays=False)

        disc_s['surface_rectangles']['numSR_x'] = 0
        self.assertRaises(ValueError, self.satDisc.discretize_outer_surface, solar_s, disc_s)

        disc_s['surface_rectangles']['numSR_x'] = 1
        disc_s['surface_rectangles']['numSR_x'] = -2
        self.assertRaises(ValueError, self.satDisc.discretize_outer_surface, solar_s, disc_s)


if __name__ == '__main__':
    unittest.main()
