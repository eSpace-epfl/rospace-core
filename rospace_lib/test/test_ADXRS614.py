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
from rospace_lib.sensor import *

class ADXRS614Test(unittest.TestCase):


    def test_truncation_non_saturated(self):
        sensor = ADXRS614()
        min_val = np.deg2rad(-10)
        max_val = np.deg2rad(10)

        sensor.max_rotation = max_val
        sensor.min_rotation = min_val

        # generate 1000 non-saturated samples, zero-mean random samples
        imu_input = np.random.normal(0,np.sqrt(abs(min_val)), 1000)
        imu_output = np.zeros(1000)
        for i in range(0, 1000):
            sensor.set_true_value(imu_input[i], 0)
            imu_output[i] = sensor.get_value()

        # in case of non-saturation, the expected mean error between in and output is around 0
        self.assertAlmostEqual(np.mean(imu_output - imu_input), 0, 1)
        self.assertNotAlmostEqual(np.mean(imu_output - max_val), 0, 1)
        self.assertNotAlmostEqual(np.mean(imu_output - min_val), 0, 1)

    def test_truncation_saturated_max(self):
        sensor = ADXRS614()
        min_val = np.deg2rad(-10)
        max_val = np.deg2rad(10)

        sensor.max_rotation = max_val
        sensor.min_rotation = min_val

        # generate 1000 non-saturated samples, zero-mean random samples
        imu_input = np.random.normal(15,np.sqrt(abs(min_val)), 1000)
        imu_output = np.zeros(1000)
        for i in range(0, 1000):
            sensor.set_true_value(imu_input[i], 0)
            imu_output[i] = sensor.get_value()

        # in case of non-saturation, the expected mean error between max_val and output is around 0
        self.assertNotAlmostEqual(np.mean(imu_output - imu_input), 0, 1)
        self.assertNotAlmostEqual(np.mean(imu_output - min_val), 0, 1)
        self.assertAlmostEqual(np.mean(imu_output - max_val), 0, 1)


    def test_truncation_saturated_min(self):
        sensor = ADXRS614()
        min_val = np.deg2rad(-10)
        max_val = np.deg2rad(10)

        sensor.max_rotation = max_val
        sensor.min_rotation = min_val

        # generate 1000 non-saturated samples, zero-mean random samples
        imu_input = np.random.normal(-55,np.sqrt(abs(min_val)), 1000)
        imu_output = np.zeros(1000)
        for i in range(0, 1000):
            sensor.set_true_value(imu_input[i], 0)
            imu_output[i] = sensor.get_value()

        # in case of non-saturation, the expected mean error between max_val and output is around 0
        self.assertNotAlmostEqual(np.mean(imu_output - imu_input), 0, 1)
        self.assertNotAlmostEqual(np.mean(imu_output - max_val), 0, 1)
        self.assertAlmostEqual(np.mean(imu_output - min_val), 0, 1)




if __name__ == '__main__':
    unittest.main()

