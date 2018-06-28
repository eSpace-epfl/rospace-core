# @copyright Copyright (c) 2017, Christian Lanegger (lanegger.christian@gmail.com)
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
import pkg_resources

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...

from rospace_lib.sensor import *


class DTUMOEMSSTest(unittest.TestCase):

    @classmethod
    def setUp(self):
        """Loads sun sensor characteristics from table and sets up sensor model."""

        # find relative path
        # pkg_resource returns path to rospace_lib package not the folder therefore return 2 levels up
        sunsensor_path = pkg_resources.resource_filename('rospace_lib', '../../doc/sunsensor/sunsensor.csv')
        sunsensor_sdev_path = pkg_resources.resource_filename('rospace_lib', '../../doc/sunsensor/sunsensor_sdev.csv')

        self.sensor = DTUMOEMSS(np.identity(4))

        # read csv and set sensor functions
        data_trafo = np.genfromtxt(sunsensor_path,
                                   delimiter=",",
                                   skip_header=1)
        self.sensor.set_transfer_func(np.deg2rad(data_trafo[:,0]), data_trafo[:,1])

        # calculate error over all samples
        samples_y_calc = np.zeros(data_trafo[:,1].shape)
        for i in range(0, len(data_trafo[:,0])):
            samples_y_calc[i] = self.sensor.transfer_func(np.deg2rad(data_trafo[i,0]))
        self.transfer_error = np.linalg.norm(data_trafo[:,1] - samples_y_calc)

        data_sdev = np.genfromtxt(sunsensor_sdev_path,
                                  delimiter=",",
                                  skip_header=1)
        self.sensor.set_noise_func(np.deg2rad(data_sdev[:,0]), data_sdev[:,1])

        # calculate error over all samples
        samples_y_calc = np.zeros(data_sdev[:,1].shape)
        for i in range(0, len(data_sdev[:,0])):
            samples_y_calc[i] = self.sensor.noise_func(np.deg2rad(data_sdev[i,0]))
        self.noise_error = np.linalg.norm(data_sdev[:,1] - samples_y_calc)

    def test_out_of_bounds_angles(self):
        """Test if NaN is returned if any angle of the sensor is larger than field of view.of

        1000 random samples are generated for both angles of the two-xis sun sensor. At least
        one of the angles is not in the field of view of the sensor.
        """
        sensor_min = np.deg2rad(-30)
        sensor_max = np.deg2rad(30)
        self.sensor.max_val = sensor_max
        self.sensor.min_val = sensor_min

        # create randomized angle samples which are out-of-bound of min, max values and have mean 60
        sv_input = np.zeros(2000).reshape(1000, 2)
        sv_output = np.zeros(2000).reshape(1000, 2)

        input_angle_x = np.random.normal(60, np.sqrt(abs(sensor_max)), 1000)
        input_angle_y = np.random.normal(0, np.sqrt(abs(sensor_max)), 1000)

        # make sure at least angle of first sensor of samples is out of bounds:
        input_angle_x[input_angle_x > sensor_max] = sensor_max + (sensor_max * 0.01)
        input_angle_x[input_angle_x < sensor_min] = sensor_min - (sensor_max * 0.01)

        # convert angle to vector and set z = 1
        sv_input[:, 0] = np.tan(input_angle_x)
        sv_input[:, 1] = np.tan(input_angle_y)

        for i in xrange(0, 1000):
            self.sensor.set_true_value(np.append(sv_input[i], 1))
            sv_output[i] = self.sensor.get_value()

        self.assertTrue(np.isnan(sv_output).all())

    def test_transfer_func(self):
        """Test transfer function and sensor output.sensor

        1000 random samples for both angles of the two-axis sun sensor are generated. Those
        angles are smaller than the maximal angle. The expected mean error between in and
        output is around 0.
        """
        # set min/max values to +-60 degrees
        sensor_min = np.deg2rad(-60)
        sensor_max = np.deg2rad(60)
        self.sensor.max_val = sensor_max
        self.sensor.min_val = sensor_min

        # create randomized angle samples which are in bound of min, max values and have mean zero
        sv_input = np.zeros(2000).reshape(1000, 2)
        sv_output = np.zeros(2000).reshape(1000, 2)

        input_angle_x = np.random.normal(0, np.sqrt(abs(sensor_max)), 1000)
        input_angle_y = np.random.normal(0, np.sqrt(abs(sensor_max)), 1000)

        # limit input angle to maximum:
        input_angle_x[input_angle_x > sensor_max] = sensor_max - (sensor_max * 0.01)
        input_angle_x[input_angle_x < sensor_min] = sensor_min + (sensor_max * 0.01)
        input_angle_y[input_angle_y > sensor_max] = sensor_max - (sensor_max * 0.01)
        input_angle_y[input_angle_y < sensor_min] = sensor_min + (sensor_max * 0.01)

        # convert angle to vector and set z = 1
        sv_input[:, 0] = np.tan(input_angle_x)
        sv_input[:, 1] = np.tan(input_angle_y)

        for i in xrange(0, 1000):
            self.sensor.set_true_value(np.append(sv_input[i], 1))
            sv_output[i] = self.sensor.get_value()

        # in case of non-saturation, the expected mean error between in and output is around 0
        self.assertAlmostEqual(np.mean(sv_output - sv_input), 0, 1)


if __name__ == '__main__':
    unittest.main()
