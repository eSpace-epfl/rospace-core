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

    def test_transfer_func(self):
        sensor = DTUMOEMSS()

        # find relative path
        # pkg_resource returns path to rospace_lib package not the folder therefore return 2 levels up
        sunsensor_path = pkg_resources.resource_filename('rospace_lib', '../../doc/sunsensor/sunsensor.csv')
        sunsensor_sdev_path = pkg_resources.resource_filename('rospace_lib', '../../doc/sunsensor/sunsensor_sdev.csv')

        # read csv
        data = np.genfromtxt(sunsensor_path,
                             delimiter=",",
                             skip_header=1)

        sensor.set_transfer_func(np.deg2rad(data[:,0]), data[:,1])

        data_sdev = np.genfromtxt(sunsensor_sdev_path,
                                  delimiter=",",
                                  skip_header=1)

        sensor.set_noise_func(np.deg2rad(data_sdev[:,0]), data_sdev[:,1])

        # sun vector
        vec_s = np.array([-2000, 0, 1000])
        sensor.set_true_value(vec_s)

        print sensor.get_value()


if __name__ == '__main__':
    unittest.main()
