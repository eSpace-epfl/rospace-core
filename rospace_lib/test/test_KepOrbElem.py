# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

# Note: these tests are quite preliminary....

import unittest
import sys
import os
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
from rospace_lib import *

class KepOrbElemTest(unittest.TestCase):

    def test_precision_chain(self):
        """
        This function tests the precision of conversion functions in a
        probabilistic way.

        A sampling (n=10000) of orbits are taken:
          - a: Uniform between 6700 and 90000 km
          - e: Uniform between exp(0) and exp(-20)
          - O,i,w,v: Uniform between 0..2Pi

        For each sampling, the following calculations are performed:
          - Get state vector p_source
          - Perturb state vector by a few mm/cm to obtain p_source_per (in order to obtain numerical non-optimal cases)
          - Convert back to orbital elements
          - Assign to new orbital element object, but use mean-anomaly
          - Convert back to perturbed state vector using true anomaly
          - Compare to original perturbed state vector and calculate deviance

        Altough a bit a convoluted method, it allows to test
        the precision of the Cartesian-Keplerian mapping and the
        True Anomaly-Mean Anomaly mapping in absence of ground truth data.

        """
        km = 1
        m = km/1e3
        cm = km/1e5
        mm = km/1e6
        um = km/1e9
        max_err = um*100  # set 100 um max error after conversion

        num_tests = int(1e4)
        a_min = 6700
        a_max = 90000
        e_min = 0.0
        e_max = -20

        err_samples = np.zeros([num_tests])

        rad_min = 0.0
        rad_max = np.pi*1.999

        err_long = []

        perturb_min = mm
        perturb_max = cm
        random_max = np.array([a_max, e_max, rad_max, rad_max, rad_max, rad_max])
        random_min = np.array([a_min, e_min, rad_min, rad_min, rad_min, rad_min])
        random_scaling = random_max - random_min

        # perform random tests
        for i in range(1, num_tests):
            random_vector = random_scaling * np.random.random_sample([6]) + random_min

            # # generate orbital element object
            O_source = KepOrbElem()
            O_source.a = random_vector[0]
            O_source.e = np.exp(random_vector[1])
            O_source.O = random_vector[2]
            O_source.w = random_vector[3]
            O_source.i = random_vector[4]
            O_source.v = random_vector[5]

            # generate status vector
            p_source = Cartesian()
            p_source.from_keporb(O_source)
            r_source = p_source.R  # [km]
            v_source = p_source.V  # [km]

            # perturb these numbers a bit
            r_source_per = r_source + (perturb_max - perturb_min)*np.random.random_sample([3])+ perturb_min
            v_source_per = v_source + (perturb_max - perturb_min)*np.random.random_sample([3])+ perturb_min


            # Generate cartesian object with perturbed numbers
            p_source_per = Cartesian()
            p_source_per.R = r_source_per
            p_source_per.V = v_source_per


            # convert to orbital elements
            O_X = KepOrbElem()
            O_X.from_cartesian(p_source_per)

            # convert back
            p_source_per_2 = Cartesian()
            p_source_per_2.from_keporb(O_X)

            # convert to orbital element with different anomaly
            O_X_2 = KepOrbElem()
            O_X_2.a = O_X.a
            O_X_2.e = O_X.e
            O_X_2.i = O_X.i
            O_X_2.w = O_X.w
            O_X_2.O = O_X.O
            O_X_2.m = O_X.m

            # convert back v2
            p_target = Cartesian()
            p_target.from_keporb(O_X_2)

            # compare
            p_err = abs(p_target.R -p_source_per.R)
            err_samples[i-1] = np.max(p_err)

            if(err_samples[i-1] > m):
                print O_X.a, O_X.e, np.rad2deg(O_X.i),np.rad2deg(O_X.w),np.rad2deg(O_X.O),np.rad2deg(O_X.v)
                print np.linalg.norm(p_target.R - p_source_per.R)
                print np.linalg.norm(p_target.V - p_source_per.V)
                print np.linalg.norm(p_source_per.R - p_source_per_2.R)
                print np.linalg.norm(p_source_per.V - p_source_per_2.V)
                print np.linalg.norm(p_target.R - p_source_per_2.R)
                print np.linalg.norm(p_target.V - p_source_per_2.V)
                print (O_X.a - O_source.a),(O_X.e - O_source.e),(O_X.i - O_source.i),(O_X.w - O_source.w),(O_X.O - O_source.O),(O_X.v - O_source.v)

            if i % 10000 == 0:
                print i

        # assign....
        percent_um = np.sum(err_samples<=um)/float(num_tests)*100.0
        percent_mm = np.sum(err_samples <= mm) / float(num_tests) * 100.0
        percent_cm =  np.sum(err_samples <= cm) / float(num_tests) * 100.0
        percent_m = np.sum(err_samples <= m) / float(num_tests) * 100.0
        percent_max_err = np.sum(err_samples <= max_err) / float(num_tests) * 100.0

        print ""
        print "Test statistics (n=", num_tests,")"
        print "===================="
        print "Max dev\t Percent pass"
        print "1 um:\t", percent_um, "%"
        print "1 mm:\t", percent_mm, "%"
        print "1 cm:\t", percent_cm, "%"
        print "1 m:\t", percent_m, "%"

        print "100um: \t", percent_max_err, "%"
        # 99.9% have to be smaller than max_err
        # 99.0% have to be smaller than 1 mm
        self.assertTrue(percent_max_err >= 99.9)
        self.assertTrue(percent_mm >= 99.0)

    def test_true_to_mean_anomaly(self):
        """
        Tests true to mean anomaly conversion

        """

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
        """
        Tests mean to true anomaly conversion
        """

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
        """
        Tests true to eccentric anomaly conversion

        """
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
        """
        Tests mean to eccentric anomaly conversion
        """
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

