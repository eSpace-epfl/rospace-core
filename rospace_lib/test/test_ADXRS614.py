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
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...

from rospace_lib.sensor import *


class ADXRS614Test(unittest.TestCase):

    def test_truncation_non_saturated(self):
        """Test sensor readings of the angular rate if values are not saturated.

        Test generates 1000 non-saturated, zero-mean random samples. The expected
        mean error between in and output is around 0.
        """
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
        """Test sensor readings of the angular rate if values saturated at the maximal value.

        Test generates 1000 saturated random samples with a mean of 15. The expected
        mean error between maximal rotation and output is around 0.
        """
        sensor = ADXRS614()
        min_val = np.deg2rad(-10)
        max_val = np.deg2rad(10)

        sensor.max_rotation = max_val
        sensor.min_rotation = min_val

        # generate 1000 saturated random samples with mean of 15
        imu_input = np.random.normal(15, np.sqrt(abs(min_val)), 1000)
        imu_output = np.zeros(1000)
        for i in range(0, 1000):
            sensor.set_true_value(imu_input[i], 0)
            imu_output[i] = sensor.get_value()

        self.assertNotAlmostEqual(np.mean(imu_output - imu_input), 0, 1)
        self.assertNotAlmostEqual(np.mean(imu_output - min_val), 0, 1)
        self.assertAlmostEqual(np.mean(imu_output - max_val), 0, 1)

    def test_truncation_saturated_min(self):
        """Test sensor readings of the angular rate if values saturated at the minimal value.

        Test generates 1000 saturated random samples with a mean of -15. The expected
        mean error between minimal rotation and output is around 0.
        """
        sensor = ADXRS614()
        min_val = np.deg2rad(-10)
        max_val = np.deg2rad(10)

        sensor.max_rotation = max_val
        sensor.min_rotation = min_val

        # generate 1000 saturated random samples with a mean of -55
        imu_input = np.random.normal(-15,np.sqrt(abs(min_val)), 1000)
        imu_output = np.zeros(1000)
        for i in range(0, 1000):
            sensor.set_true_value(imu_input[i], 0)
            imu_output[i] = sensor.get_value()

        self.assertNotAlmostEqual(np.mean(imu_output - imu_input), 0, 1)
        self.assertNotAlmostEqual(np.mean(imu_output - max_val), 0, 1)
        self.assertAlmostEqual(np.mean(imu_output - min_val), 0, 1)


if __name__ == '__main__':
    unittest.main()
